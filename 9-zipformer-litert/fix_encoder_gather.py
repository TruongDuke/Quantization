"""Thay GatherElements trong encoder bằng Reshape + Gather (để LiteRT chạy được).

Sau khi cố định batch = 1 và chạy onnxsim, cả 10 GatherElements (trong phần mã hoá vị trí tương đối của attention)
đều có data 2 chiều, gather theo chiều 1, indices là hằng số:
    out[i, j] = data[i, idx[i, j]]
Tương đương: duỗi data thành 1 chiều rồi Gather với chỉ số phẳng (tính sẵn) i * so_cot + idx[i, j].

Backend flatbuffer_direct của onnx2tf lower GatherElements bị sai shape: lúc convert không báo lỗi,
nhưng file .tflite báo lỗi CONCATENATION (128 != 1) khi chạy.
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

SRC = "model_onnx/encoder.b1.sim.onnx"          # đã cố định batch = 1 và chạy onnxsim
DST = "model_onnx/encoder.b1.sim.fixgather.onnx"
ORIG = "model_onnx/encoder-epoch-99-avg-1.onnx"

m = onnx.load(SRC)
g = m.graph
init = {t.name: t for t in g.initializer}
inferred = onnx.shape_inference.infer_shapes(m)
shape = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
         for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)}

count = 0
for node in [n for n in g.node if n.op_type == "GatherElements"]:
    data, idx_name = node.input
    axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
    idx = numpy_helper.to_array(init[idx_name]).astype(np.int64)
    dshape = shape[data]
    assert len(dshape) == 2 and axis == 1 and idx.ndim == 2, (node.name, dshape, axis, idx.shape)
    rows, cols = dshape
    idx = np.where(idx < 0, idx + cols, idx)
    flat_idx = (np.arange(idx.shape[0])[:, None] * cols + idx).astype(np.int64)
    p = node.name.strip("/").replace("/", "_")
    g.initializer.extend([numpy_helper.from_array(np.array([-1], np.int64), f"{p}_flat_shape"),
                          numpy_helper.from_array(flat_idx, f"{p}_flat_idx")])
    pos = list(g.node).index(node)
    g.node.remove(node)
    g.node.insert(pos, helper.make_node("Reshape", [data, f"{p}_flat_shape"], [f"{p}_flat"]))
    g.node.insert(pos + 1, helper.make_node("Gather", [f"{p}_flat", f"{p}_flat_idx"], [node.output[0]], axis=0))
    count += 1
for name in list(init):
    if not any(name in n.input for n in g.node):
        g.initializer.remove(init[name])
onnx.checker.check_model(m)
onnx.save(m, DST)

# Kiểm tra: so với model ONNX gốc trên vài chunk ngẫu nhiên, trạng thái đưa vòng lại như khi chạy thật
a, b = ort.InferenceSession(ORIG), ort.InferenceSession(DST)
spec = {i.name: ([1 if isinstance(d, str) else d for d in i.shape], np.int64 if "int64" in i.type else np.float32) for i in a.get_inputs()}
sa = {n: np.zeros(s, t) for n, (s, t) in spec.items() if n != "x"}
sb = dict(sa)
rng = np.random.default_rng(0)
err = 0.0
for step in range(5):
    x = rng.standard_normal(spec["x"][0]).astype(np.float32)
    oa = dict(zip([o.name for o in a.get_outputs()], a.run(None, {"x": x, **sa})))
    ob = dict(zip([o.name for o in b.get_outputs()], b.run(None, {"x": x, **sb})))
    err = max(err, float(np.abs(oa["encoder_out"] - ob["encoder_out"]).max()))
    sa = {n: oa["new_" + n] for n in sa}
    sb = {n: ob["new_" + n] for n in sb}
print(f"Thay {count} GatherElements. Da luu {DST}. Lech encoder_out so voi goc qua 5 chunk: {err:.2e}")
