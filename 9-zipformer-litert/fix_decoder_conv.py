"""Thay grouped Conv1d trong decoder bằng MatMul tương đương (để LiteRT convert được).

Decoder: Conv1d(512 -> 512, kernel=2, group=128) trên input (1, 512, 2) = 2 token gần nhất.
Kernel bằng đúng độ dài input, không padding -> output chỉ có 1 vị trí, nên:
    out[o] = bias[o] + sum_{c thuộc nhóm của o, t} W[o, c - 4*g, t] * x[c, t]
tức là nhân vector x (1024 số) với ma trận 1024 x 512 dạng khối chéo.

onnx2tf không convert được op này: backend flatbuffer_direct để lại custom op ONNX_CONV (runtime không chạy được),
backend tf_converter thì đọc nhầm số kênh (lỗi "input has 2 channels").
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

SRC = "model_onnx/decoder-epoch-99-avg-1.onnx"
DST = "model_onnx/decoder-epoch-99-avg-1.fixconv.onnx"

m = onnx.load(SRC)
g = m.graph
init = {t.name: t for t in g.initializer}
conv = next(n for n in g.node if n.op_type == "Conv")
attrs = {a.name: helper.get_attribute_value(a) for a in conv.attribute}
W = numpy_helper.to_array(init[conv.input[1]])                 # (512, 4, 2)
b = numpy_helper.to_array(init[conv.input[2]]) if len(conv.input) > 2 else np.zeros(W.shape[0], np.float32)
groups, (out_ch, in_per_g, k) = attrs["group"], W.shape
out_per_g = out_ch // groups
in_ch = in_per_g * groups
assert attrs.get("pads", [0, 0]) == [0, 0] and k == 2

dense = np.zeros((in_ch * k, out_ch), np.float32)               # hàng = c * k + t (khớp với reshape (512, 2) -> 1024)
for o in range(out_ch):
    gi = o // out_per_g
    for ci in range(in_per_g):
        c = gi * in_per_g + ci
        dense[c * k:(c + 1) * k, o] = W[o, ci, :]

x, y = conv.input[0], conv.output[0]
p = conv.name.replace("/", "_")
g.initializer.extend([
    numpy_helper.from_array(dense, f"{p}_dense"),
    numpy_helper.from_array(b.astype(np.float32), f"{p}_bias"),
    numpy_helper.from_array(np.array([1, in_ch * k], np.int64), f"{p}_shape_in"),
    numpy_helper.from_array(np.array([1, out_ch, 1], np.int64), f"{p}_shape_out"),
])
new_nodes = [
    helper.make_node("Reshape", [x, f"{p}_shape_in"], [f"{p}_flat"]),
    helper.make_node("MatMul", [f"{p}_flat", f"{p}_dense"], [f"{p}_mm"]),
    helper.make_node("Add", [f"{p}_mm", f"{p}_bias"], [f"{p}_add"]),
    helper.make_node("Reshape", [f"{p}_add", f"{p}_shape_out"], [y]),
]
idx = list(g.node).index(conv)
g.node.remove(conv)
for i, n in enumerate(new_nodes):
    g.node.insert(idx + i, n)
for name in conv.input[1:]:                                      # bỏ weight cũ nếu không còn dùng
    if name in init and not any(name in n.input for n in g.node):
        g.initializer.remove(init[name])
for v in list(g.input) + list(g.output):                         # batch cố định = 1
    d = v.type.tensor_type.shape.dim[0]
    if d.dim_param:
        d.ClearField("dim_param"); d.dim_value = 1
onnx.checker.check_model(m)
onnx.save(m, DST)

# Kiểm tra: output của model sửa phải khớp model gốc
a, bsess = ort.InferenceSession(SRC), ort.InferenceSession(DST)
err = 0.0
for _ in range(200):
    yy = np.random.randint(0, 500, size=(1, 2)).astype(np.int64)
    err = max(err, float(np.abs(a.run(None, {"y": yy})[0] - bsess.run(None, {"y": yy})[0]).max()))
print(f"Da luu {DST}. Lech lon nhat so voi goc tren 200 input ngau nhien: {err:.2e}")
