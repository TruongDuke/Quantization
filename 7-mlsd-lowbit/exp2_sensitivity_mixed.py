"""Thí nghiệm 2 (dựa trên model tách head, static INT8):
(a) Độ nhạy từng layer: mỗi lần chỉ quantize 1 layer conv, các layer khác giữ float.
(b) Mixed precision: giữ N layer nhạy nhất ở float, còn lại INT8.
(c) Đổi thuật toán chọn khoảng cho weight: MSE, OCTAV thay MinMax.
Dùng lại dữ liệu, metric và hàm của exp1 (không chạy lại phần thí nghiệm của exp1).
"""
import os
import re
import json
import copy
import numpy as np
from ai_edge_quantizer import quantizer, recipe, algorithm_manager
from ai_edge_litert.interpreter import Interpreter

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp1_metrics_and_heads.py")).read()
exec(src[:src.index("rows = []")])            # nạp dữ liệu, model, metric, REF (không chạy bảng của exp1)

FP32 = f"{MODELS}/split_fp32.tflite"
NO_Q = algorithm_manager.AlgorithmName.NO_QUANTIZE

# ----- Danh sách layer conv và tên tensor output (dùng làm regex chính xác)
it = Interpreter(FP32)
it.allocate_tensors()
td = it.get_tensor_details()
convs = []
for o in it._get_ops_details():
    if o["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D"):
        full = td[o["outputs"][0]]["name"]
        short = full.split(";;")[-1].replace("__main__.Split/", "").replace("models.mbv2_mlsd_tiny.", "") \
                    .replace("torch.nn.modules.", "").replace("container.Sequential_", "").replace(";2", "")
        convs.append({"op": o["index"], "type": o["op_name"][:5], "regex": "^" + re.escape(full) + ";$", "short": short})
print("So layer conv:", len(convs))

# ----- Calibration 1 lần, dùng lại cho mọi recipe
qt = quantizer.Quantizer(FP32, recipe.static_wi8_ai8())
sig_name, sig = list(Interpreter(FP32).get_signature_list().items())[0]
CALIB = qt.calibrate({sig_name: [{sig["inputs"][0]: calib_x[i:i + 1]} for i in range(len(calib_x))]})
STATIC = recipe.static_wi8_ai8()[0]

def no_q(rx):
    return {"regex": rx, "operation": "*", "algorithm_key": NO_Q, "op_config": {}}

def build(rcp, path):
    q = quantizer.Quantizer(FP32, rcp)
    q.quantize(CALIB).export_model(path, overwrite=True)
    return path

def score(path):
    r = TFLiteRunner(path)
    maps = [from_split(r(x)) for x in eval_x]
    return evaluate(lambda i: maps[i])

os.makedirs(f"{MODELS}/tmp", exist_ok=True)
tmp = f"{MODELS}/tmp/m.tflite"

# ===== (a) Độ nhạy: chỉ quantize layer k =====
sens = []
for k, c in enumerate(convs):
    rcp = [no_q(".*"), {**STATIC, "regex": c["regex"]}]
    m = score(build(rcp, tmp))
    sens.append({**c, **m})
    print(f"[a] {k:2d} {c['type']} {c['short'][:60]:60s} giu {m['lines_kept']:5.1f}  center {m['center_recall']:5.1f}  Dmed {m['disp_err_median_px']:.2f}")
order = sorted(range(len(sens)), key=lambda i: sens[i]["lines_kept"])     # layer làm hỏng nhiều nhất trước

# ===== (b) Mixed precision: giữ N layer nhạy nhất ở float =====
mixed = []
for n in [0, 1, 2, 3, 5, 8, 12]:
    rcp = [copy.deepcopy(STATIC)] + [no_q(sens[i]["regex"]) for i in order[:n]]
    path = f"{MODELS}/split_mixed_float{n}.tflite"
    m = score(build(rcp, path))
    mixed.append({"n_float": n, "size_mb": os.path.getsize(path) / 1e6, "ms": benchmark(path), **m})
    print(f"[b] float {n:2d} layer: giu {m['lines_kept']:5.1f} center {m['center_recall']:5.1f} Dmed {m['disp_err_median_px']:.2f} "
          f"| {os.path.getsize(path)/1e6:.2f} MB {mixed[-1]['ms']:.1f} ms")

# ===== (c) Thuật toán chọn khoảng cho weight =====
algo = []
for key in ["min_max_uniform_quantize", "MSE", "OCTAV"]:
    rcp = [{**copy.deepcopy(STATIC), "algorithm_key": key}]
    try:
        q = quantizer.Quantizer(FP32, rcp)
        cal = q.calibrate({sig_name: [{sig["inputs"][0]: calib_x[i:i + 1]} for i in range(len(calib_x))]})
        path = f"{MODELS}/split_static_{key}.tflite"
        q.quantize(cal).export_model(path, overwrite=True)
        m = score(path)
        algo.append({"algorithm": key, **m})
        print(f"[c] {key:26s} giu {m['lines_kept']:5.1f} center {m['center_recall']:5.1f} Dmed {m['disp_err_median_px']:.2f}")
    except Exception as e:
        algo.append({"algorithm": key, "error": f"{type(e).__name__}: {str(e)[:150]}"})
        print(f"[c] {key:26s} LOI: {type(e).__name__}: {str(e)[:150]}")

with open(f"{ROOT}/results/exp2.json", "w") as f:
    json.dump({"sensitivity": sens, "mixed": mixed, "algorithm": algo}, f, indent=1)
