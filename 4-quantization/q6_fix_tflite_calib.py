import numpy as np
from ai_edge_quantizer import quantizer, recipe, algorithm_manager
from ai_edge_litert.interpreter import Interpreter

val_x = np.load("val_x.npy")
ref = np.load("ref_output.npy")
calib_x = np.load("calib_x.npy")

# ===== Hàm min/max toàn cục (giống ONNX Runtime) =====
def global_min_max_update(qsv, new_qsv):
    if not qsv:                                   # ảnh đầu tiên: chưa có gì, lấy luôn
        return new_qsv
    return {"min": np.minimum(qsv["min"], new_qsv["min"]),
            "max": np.maximum(qsv["max"], new_qsv["max"])}

original_func = algorithm_manager.get_update_qsv_func      # cất hàm gốc (EMA) để dùng lại

def use_global_min_max(on):
    if on:
        algorithm_manager.get_update_qsv_func = lambda *args: global_min_max_update
    else:
        algorithm_manager.get_update_qsv_func = original_func

# ===== Quantize và đánh giá TFLite static =====
def quant_tflite_static(data, out_path):
    qt = quantizer.Quantizer("resnet18_fp32.tflite", recipe.static_wi8_ai8())
    calib_data = {"serving_default": [{"args_0": data[i:i+1]} for i in range(len(data))]}
    qt.quantize(qt.calibrate(calib_data)).export_model(out_path, overwrite=True)

def evaluate(path):
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
    outs = np.concatenate(outs)
    lo, hi = (-128 - zp_out) * s_out, (127 - zp_out) * s_out
    agree = np.mean(outs.argmax(1) == ref.argmax(1)) * 100
    snr = 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - outs) ** 2))
    return lo, hi, agree, snr

# ===== So sánh: EMA (mặc định) và min/max toàn cục, với 2 thứ tự ảnh =====
orders = {"theo_class": calib_x, "dao_nguoc": calib_x[::-1]}
rows = []
for mode in ["EMA (mac dinh)", "Min/max toan cuc"]:
    use_global_min_max(mode != "EMA (mac dinh)")
    for order_name, data in orders.items():
        path = f"calib_exp/D_{'global' if mode != 'EMA (mac dinh)' else 'ema'}_{order_name}.tflite"
        quant_tflite_static(data, path)
        rows.append((mode, order_name, *evaluate(path)))
use_global_min_max(False)

print(f"\n{'Cach tinh':18s} {'Thu tu':11s} {'Khoang output':>16s} {'Agree(%)':>9s} {'SNR(dB)':>8s}")
for mode, order_name, lo, hi, agree, snr in rows:
    print(f"{mode:18s} {order_name:11s} {f'[{lo:.1f}, {hi:.1f}]':>16s} {agree:9.1f} {snr:8.1f}")
