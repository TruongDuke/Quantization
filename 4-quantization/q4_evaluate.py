import os
import time
import numpy as np
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter

val_x = np.load("val_x.npy")
val_y = np.load("val_y.npy")
ref = np.load("ref_output.npy")   # output PyTorch gốc

# ===== Tạo hàm chạy cho từng loại model =====
def make_onnx_runner(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    return lambda x: sess.run(None, {"input": x})[0]

def make_tflite_runner(path):
    interp = Interpreter(model_path=path, num_threads=4)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    def run(x):
        if inp["dtype"] == np.int8:                      # model nhận INT8 → tự quantize input
            scale, zp = inp["quantization"]
            x = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])
        if out["dtype"] == np.int8:                      # model trả INT8 → dequantize output
            scale, zp = out["quantization"]
            y = (y.astype(np.float32) - zp) * scale
        return y
    return run

# ===== Chạy 300 ảnh và tính metric =====
def evaluate(run):
    outs = np.concatenate([run(val_x[i:i+1]) for i in range(len(val_x))])
    acc = np.mean(outs.argmax(1) == val_y) * 100
    agree = np.mean(outs.argmax(1) == ref.argmax(1)) * 100
    snr = 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - outs) ** 2))
    return acc, agree, snr

def benchmark(run, warmup=10, runs=100):
    x = val_x[:1]
    for _ in range(warmup):
        run(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run(x)
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times)

MODELS = [
    ("ONNX FP32",      "resnet18_fp32.onnx",             make_onnx_runner),
    ("ONNX dynamic",   "resnet18_onnx_dynamic.onnx",     make_onnx_runner),
    ("ONNX static",    "resnet18_onnx_static.onnx",      make_onnx_runner),
    ("TFLite FP32",    "resnet18_fp32.tflite",           make_tflite_runner),
    ("TFLite dynamic", "resnet18_tflite_dynamic.tflite", make_tflite_runner),
    ("TFLite static",  "resnet18_tflite_static.tflite",  make_tflite_runner),
]

print(f"PyTorch gốc: accuracy = {np.mean(ref.argmax(1) == val_y) * 100:.1f}%\n")
print(f"{'Model':16s} {'Size(MB)':>9s} {'Acc(%)':>7s} {'Agree(%)':>9s} {'SNR(dB)':>8s} {'Time(ms)':>9s}")
for name, path, make_runner in MODELS:
    run = make_runner(path)
    acc, agree, snr = evaluate(run)
    t = benchmark(run)
    size = os.path.getsize(path) / 1e6
    print(f"{name:16s} {size:9.1f} {acc:7.1f} {agree:9.1f} {snr:8.1f} {t:9.2f}")
