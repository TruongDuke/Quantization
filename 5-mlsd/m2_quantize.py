import os
import sys
import glob
import time
import numpy as np
import cv2
import torch
from ai_edge_quantizer import quantizer, recipe, algorithm_manager
from ai_edge_litert.interpreter import Interpreter

sys.path.insert(0, "mlsd-pytorch")
from utils import pred_lines as repo_pred_lines   # hàm post-processing của repo PyTorch

def pred_lines(img, model, input_shape, score_thr, dist_thr):
    # Hàm của repo bị lỗi khi không tìm thấy đoạn thẳng nào (mảng rỗng), nên bọc lại
    try:
        return repo_pred_lines(img, model, input_shape, score_thr, dist_thr)
    except IndexError:
        return np.zeros((0, 4))

FP32 = "mlsd_tiny_raw.tflite"
IMAGENETTE = "../4-quantization/imagenette2-160"

# ===== Tiền xử lý giống pred_lines: resize 512, thêm kênh toàn số 1, chuẩn hoá [-1, 1], NCHW =====
def preprocess(img_rgb):
    img = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
    x = np.concatenate([img, np.ones([512, 512, 1])], axis=-1).transpose(2, 0, 1)[None]
    return (x.astype(np.float32) / 127.5) - 1.0

def read_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

# ===== Ảnh calibration (100 ảnh train) và ảnh đánh giá (frame_1 + 20 ảnh val) =====
train_paths = sorted(glob.glob(f"{IMAGENETTE}/train/*/*.JPEG"))
val_paths = sorted(glob.glob(f"{IMAGENETTE}/val/*/*.JPEG"))
calib_paths = np.random.default_rng(0).choice(train_paths, 100, replace=False)
eval_paths = ["mlsd-pytorch/data/frame_1.jpg"] + list(np.random.default_rng(1).choice(val_paths, 20, replace=False))
calib_x = np.concatenate([preprocess(read_rgb(p)) for p in calib_paths])
eval_imgs = [cv2.resize(read_rgb(p), (512, 512), interpolation=cv2.INTER_AREA) for p in eval_paths]
print("Calibration:", calib_x.shape, "| So anh danh gia:", len(eval_imgs))

# ===== Quantize =====
def global_min_max_update(qsv, new_qsv):          # cách sửa từ q6: min/max toàn cục thay cho EMA
    if not qsv:
        return new_qsv
    return {"min": np.minimum(qsv["min"], new_qsv["min"]),
            "max": np.maximum(qsv["max"], new_qsv["max"])}

original_update = algorithm_manager.get_update_qsv_func
calib_data = {"serving_default": [{"args_0": calib_x[i:i+1]} for i in range(len(calib_x))]}

qt = quantizer.Quantizer(FP32, recipe.dynamic_wi8_afp32())
qt.quantize().export_model("mlsd_tiny_dynamic.tflite", overwrite=True)

qt = quantizer.Quantizer(FP32, recipe.static_wi8_ai8())
qt.quantize(qt.calibrate(calib_data)).export_model("mlsd_tiny_static_ema.tflite", overwrite=True)

algorithm_manager.get_update_qsv_func = lambda *args: global_min_max_update
qt = quantizer.Quantizer(FP32, recipe.static_wi8_ai8())
qt.quantize(qt.calibrate(calib_data)).export_model("mlsd_tiny_static_global.tflite", overwrite=True)
algorithm_manager.get_update_qsv_func = original_update

# ===== Bọc TFLite thành "model" để pred_lines gọi được: nhận tensor NCHW, trả tensor (1,9,256,256) =====
def make_tflite_model(path, threads=4):
    it = Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    def model(batch):
        x = batch.numpy() if isinstance(batch, torch.Tensor) else batch
        if inp["dtype"] == np.int8:
            s, zp = inp["quantization"]
            x = np.clip(np.round(x / s + zp), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], x)
        it.invoke()
        y = it.get_tensor(out["index"])
        if out["dtype"] == np.int8:
            s, zp = out["quantization"]
            y = (y.astype(np.float32) - zp) * s
        return torch.from_numpy(y)
    return model

# ===== Đánh giá =====
def snr(ref, out):
    return 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - out) ** 2))

def match_ratio(ref_lines, lines, tol=5.0):
    # % đoạn thẳng của FP32 tìm được đoạn tương ứng (2 đầu mút lệch < tol pixel, xét cả 2 chiều)
    if len(ref_lines) == 0:
        return 100.0
    if len(lines) == 0:
        return 0.0
    hit = 0
    for r in ref_lines:
        d1 = np.maximum(np.hypot(*(lines[:, :2] - r[:2]).T), np.hypot(*(lines[:, 2:] - r[2:]).T))
        d2 = np.maximum(np.hypot(*(lines[:, :2] - r[2:]).T), np.hypot(*(lines[:, 2:] - r[:2]).T))
        hit += np.minimum(d1, d2).min() < tol
    return 100.0 * hit / len(ref_lines)

def benchmark(path, threads, warmup=10, runs=50):
    it = Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    d = it.get_input_details()[0]
    it.set_tensor(d["index"], np.zeros(d["shape"], dtype=d["dtype"]))
    for _ in range(warmup):
        it.invoke()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        it.invoke()
        t.append((time.perf_counter() - t0) * 1000)
    return np.median(t)

fp32_model = make_tflite_model(FP32)
ref_outputs = [fp32_model(preprocess(img)).numpy() for img in eval_imgs]
ref_lines = [pred_lines(img, fp32_model, [512, 512], 0.1, 20) for img in eval_imgs]

print(f"\n{'Model':16s} {'Size(MB)':>9s} {'SNR(dB)':>8s} {'So doan':>8s} {'Giu duoc(%)':>12s} {'1 thr(ms)':>10s} {'4 thr(ms)':>10s}")
for name, path in [("FP32", FP32),
                   ("Dynamic", "mlsd_tiny_dynamic.tflite"),
                   ("Static (EMA)", "mlsd_tiny_static_ema.tflite"),
                   ("Static (global)", "mlsd_tiny_static_global.tflite")]:
    model = make_tflite_model(path)
    outs = [model(preprocess(img)).numpy() for img in eval_imgs]
    lines = [pred_lines(img, model, [512, 512], 0.1, 20) for img in eval_imgs]
    s = snr(np.concatenate(ref_outputs), np.concatenate(outs)) if name != "FP32" else float("inf")
    n_lines = sum(len(l) for l in lines)
    kept = np.mean([match_ratio(r, l) for r, l in zip(ref_lines, lines)])
    size = os.path.getsize(path) / 1e6
    print(f"{name:16s} {size:9.2f} {s:8.1f} {n_lines:8d} {kept:12.1f} {benchmark(path, 1):10.2f} {benchmark(path, 4):10.2f}")

# Lưu ảnh kết quả trên frame_1 để nhìn bằng mắt
for name, path in [("fp32", FP32), ("static_global", "mlsd_tiny_static_global.tflite")]:
    img = eval_imgs[0].copy()
    for x1, y1, x2, y2 in pred_lines(img, make_tflite_model(path), [512, 512], 0.1, 20):
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 200), 1)
    cv2.imwrite(f"results/frame_1_{name}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
