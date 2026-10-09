import os
import glob
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from ai_edge_quantizer import quantizer, recipe
from ai_edge_litert.interpreter import Interpreter
from PIL import Image
from torchvision.models import ResNet18_Weights

os.makedirs("calib_exp", exist_ok=True)

val_x = np.load("val_x.npy")
val_y = np.load("val_y.npy")
ref = np.load("ref_output.npy")
print(f"Khoang logits that (PyTorch, 300 anh val): [{ref.min():.1f}, {ref.max():.1f}]")

# ===== Đọc ảnh (giống q1_prepare.py) =====
preprocess = ResNet18_Weights.DEFAULT.transforms()
CLASSES = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491, "n03028079": 497,
           "n03394916": 566, "n03417042": 569, "n03425413": 571, "n03445777": 574, "n03888257": 701}

def load_images(split, n_per_class):
    images, labels = [], []
    for folder, class_id in CLASSES.items():
        paths = sorted(glob.glob(f"imagenette2-160/{split}/{folder}/*.JPEG"))[:n_per_class]
        for p in paths:
            images.append(preprocess(Image.open(p).convert("RGB")).numpy())
            labels.append(class_id)
    return np.stack(images).astype(np.float32), np.array(labels)

# ===== Kho ảnh calibration: 50 ảnh/class từ train =====
if not os.path.exists("calib_pool.npy"):
    pool, _ = load_images("train", 50)
    np.save("calib_pool.npy", pool)
pool = np.load("calib_pool.npy")
print("Kho calibration:", pool.shape)

# ===== Quantize static =====
class MyCalibReader(CalibrationDataReader):
    def __init__(self, data):
        self.data = iter([{"input": data[i:i+1]} for i in range(len(data))])

    def get_next(self):
        return next(self.data, None)

def quant_onnx_static(calib_x, out_path):
    quantize_static("resnet18_prep.onnx", out_path, MyCalibReader(calib_x),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)

def quant_tflite_static(calib_x, out_path):
    qt = quantizer.Quantizer("resnet18_fp32.tflite", recipe.static_wi8_ai8())
    calib_data = {"serving_default": [{"args_0": calib_x[i:i+1]} for i in range(len(calib_x))]}
    result = qt.calibrate(calib_data)
    qt.quantize(result).export_model(out_path, overwrite=True)

# ===== Đọc khoảng output mà model INT8 biểu diễn được =====
def int8_range(scale, zp):
    return (-128 - zp) * scale, (127 - zp) * scale

def onnx_output_range(path):
    g = onnx.load(path).graph
    producer = {out: node for node in g.node for out in node.output}
    dq = producer[g.output[0].name]                  # node DequantizeLinear cuối đồ thị
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    return int8_range(float(init[dq.input[1]]), int(init[dq.input[2]]))

def tflite_output_range(path):
    out = Interpreter(model_path=path).get_output_details()[0]
    scale, zp = out["quantization"]
    return int8_range(scale, zp)

# ===== Chạy 300 ảnh val =====
def run_onnx(path):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return np.concatenate([sess.run(None, {"input": val_x[i:i+1]})[0] for i in range(len(val_x))])

def run_tflite(path):
    interp = Interpreter(model_path=path, num_threads=4)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    s_in, zp_in = inp["quantization"]
    s_out, zp_out = out["quantization"]
    outs = []
    for i in range(len(val_x)):
        x = np.clip(np.round(val_x[i:i+1] / s_in + zp_in), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])
        outs.append((y.astype(np.float32) - zp_out) * s_out)
    return np.concatenate(outs)

def metrics(outs):
    agree = np.mean(outs.argmax(1) == ref.argmax(1)) * 100
    snr = 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - outs) ** 2))
    return agree, snr

# ===== Quantize cả 2 framework với cùng 1 bộ calibration, rồi đánh giá =====
def run_both(calib_x, tag):
    onnx_path = f"calib_exp/{tag}.onnx"
    tfl_path = f"calib_exp/{tag}.tflite"
    quant_onnx_static(calib_x, onnx_path)
    quant_tflite_static(calib_x, tfl_path)
    rows = []
    for name, path, run, get_range in [("ONNX", onnx_path, run_onnx, onnx_output_range),
                                       ("TFLite", tfl_path, run_tflite, tflite_output_range)]:
        agree, snr = metrics(run(path))
        lo, hi = get_range(path)
        rows.append((tag, name, lo, hi, agree, snr))
    return rows

def print_table(title, rows):
    print(f"\n===== {title} =====")
    print(f"{'Lan chay':14s} {'Framework':9s} {'Khoang output':>18s} {'Agree(%)':>9s} {'SNR(dB)':>8s}")
    for tag, name, lo, hi, agree, snr in rows:
        print(f"{tag:14s} {name:9s} {f'[{lo:.1f}, {hi:.1f}]':>18s} {agree:9.1f} {snr:8.1f}")

# ===== Thí nghiệm B: cùng 100 ảnh, chỉ đổi thứ tự =====
calib_x = np.load("calib_x.npy")
rng = np.random.default_rng(0)
rows = []
rows += run_both(calib_x, "B_theo_class")
rows += run_both(calib_x[::-1], "B_dao_nguoc")
rows += run_both(calib_x[rng.permutation(len(calib_x))], "B_tron")
print_table("Thi nghiem B: cung 100 anh, doi thu tu", rows)

# ===== Thí nghiệm A: 5 subset ngẫu nhiên, mỗi subset 100 ảnh =====
rows = []
for seed in range(5):
    idx = np.random.default_rng(seed).choice(len(pool), 100, replace=False)
    rows += run_both(pool[idx], f"A_seed{seed}")
print_table("Thi nghiem A: 5 subset ngau nhien (100 anh)", rows)

# ===== Thí nghiệm C: đổi số lượng ảnh calibration =====
rows = []
for n in [10, 50, 100, 300]:
    idx = np.random.default_rng(42).choice(len(pool), n, replace=False)
    rows += run_both(pool[idx], f"C_{n}anh")
print_table("Thi nghiem C: so luong anh calibration", rows)
