import sys
import time
import numpy as np
import onnxruntime as ort

# Môi trường của bạn có ai_edge_litert, môi trường của Huy dùng tf.lite của TensorFlow
try:
    from ai_edge_litert.interpreter import Interpreter
    tflite_runtime = "ai-edge-litert"
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    tflite_runtime = f"tensorflow {tf.__version__}"

print(f"Python {sys.version.split()[0]} | numpy {np.__version__} | onnxruntime {ort.__version__} | TFLite: {tflite_runtime}")

val_x = np.load("val_x.npy")
ref = np.load("ref_output.npy")

def make_onnx_runner(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    return lambda x: sess.run(None, {"input": x})[0]

def make_tflite_runner(path, threads):
    interp = Interpreter(model_path=path, num_threads=threads)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    def run(x):
        if inp["dtype"] == np.int8:
            s, zp = inp["quantization"]
            x = np.clip(np.round(x / s + zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])
        if out["dtype"] == np.int8:
            s, zp = out["quantization"]
            y = (y.astype(np.float32) - zp) * s
        return y
    return run

def agreement(run):
    outs = np.concatenate([run(val_x[i:i+1]) for i in range(len(val_x))])
    return np.mean(outs.argmax(1) == ref.argmax(1)) * 100

def benchmark(run, warmup=30, runs=200):            # giống cách đo của Huy
    x = val_x[:1]
    for _ in range(warmup):
        run(x)
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run(x)
        t.append((time.perf_counter() - t0) * 1000)
    return np.median(t)

MODELS = [
    ("ONNX FP32",      "resnet18_fp32.onnx",             make_onnx_runner),
    ("ONNX dynamic",   "resnet18_onnx_dynamic.onnx",     make_onnx_runner),
    ("ONNX static",    "resnet18_onnx_static.onnx",      make_onnx_runner),
    ("TFLite FP32",    "resnet18_fp32.tflite",           make_tflite_runner),
    ("TFLite dynamic", "resnet18_tflite_dynamic.tflite", make_tflite_runner),
    ("TFLite static",  "resnet18_tflite_static.tflite",  make_tflite_runner),
]

print(f"\n{'Model':16s} {'Agree(%)':>9s} {'1 thread(ms)':>13s} {'4 thread(ms)':>13s}")
for name, path, make_runner in MODELS:
    try:
        agree = agreement(make_runner(path, 4))
        t1 = benchmark(make_runner(path, 1))
        t4 = benchmark(make_runner(path, 4))
        print(f"{name:16s} {agree:9.1f} {t1:13.2f} {t4:13.2f}")
    except Exception as e:
        print(f"{name:16s} LOI: {type(e).__name__}: {str(e)[:80]}")
