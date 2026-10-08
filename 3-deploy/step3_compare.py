import time
import numpy as np
import torch
import torchvision
import onnxruntime as ort

# ===== Hàm tính metric =====
def compare(ref, out):
    diff = ref - out
    max_abs = np.abs(diff).max()
    mean_abs = np.abs(diff).mean()
    snr = 10 * np.log10(np.sum(ref ** 2) / np.sum(diff ** 2))
    cosine = np.sum(ref * out) / (np.linalg.norm(ref) * np.linalg.norm(out))
    same_top1 = ref.argmax() == out.argmax()
    print(f"  Max abs error : {max_abs:.2e}")
    print(f"  Mean abs error: {mean_abs:.2e}")
    print(f"  SNR           : {snr:.2f} dB")
    print(f"  Cosine sim    : {cosine:.8f}")
    print(f"  Cung top-1?   : {same_top1}")

# ===== Hàm đo tốc độ =====
def benchmark(run_fn, warmup=10, runs=100):
    for _ in range(warmup):
        run_fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run_fn()
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    print(f"  Median: {np.median(times):.2f} ms | P90: {np.percentile(times, 90):.2f} ms")

x = np.load("input.npy")
ref = np.load("output.npy")

# ===== PyTorch =====
torch.set_num_threads(4)
model = torchvision.models.mobilenet_v2(weights="DEFAULT").eval()
x_t = torch.from_numpy(x)

def run_torch():
    with torch.no_grad():
        return model(x_t)

print("PyTorch speed:")
benchmark(run_torch)

# ===== ONNX Runtime =====
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
sess = ort.InferenceSession("mobilenet_v2.onnx", opts, providers=["CPUExecutionProvider"])

def run_onnx():
    return sess.run(None, {"input": x})[0]

print("\nONNX vs PyTorch:")
compare(ref, run_onnx())
print("ONNX speed:")
benchmark(run_onnx)

# ===== TFLite =====
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="mobilenet_v2.tflite", num_threads=4)
interpreter.allocate_tensors()
in_idx = interpreter.get_input_details()[0]["index"]
out_idx = interpreter.get_output_details()[0]["index"]

def run_tflite():
    interpreter.set_tensor(in_idx, x)
    interpreter.invoke()
    return interpreter.get_tensor(out_idx)

print("\nTFLite vs PyTorch:")
compare(ref, run_tflite())
print("TFLite speed:")
benchmark(run_tflite)
