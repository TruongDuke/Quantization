import os
import numpy as np
from ai_edge_quantizer import quantizer, recipe

FP32_MODEL = "resnet18_fp32.tflite"

# Dynamic quantization 
qt = quantizer.Quantizer(FP32_MODEL, recipe.dynamic_wi8_afp32())
qt.quantize().export_model("resnet18_tflite_dynamic.tflite", overwrite=True)
print("Xong dynamic")

#  Static quantization 
calib_x = np.load("calib_x.npy")
calib_data = {
    "serving_default": [{"args_0": calib_x[i:i+1]} for i in range(len(calib_x))]
}

qt = quantizer.Quantizer(FP32_MODEL, recipe.static_wi8_ai8())
calib_result = qt.calibrate(calib_data)
qt.quantize(calib_result).export_model("resnet18_tflite_static.tflite", overwrite=True)
print("Xong static")

# So sánh kích thước =====
for f in [FP32_MODEL, "resnet18_tflite_dynamic.tflite", "resnet18_tflite_static.tflite"]:
    print(f"{f:32s} {os.path.getsize(f) / 1e6:.1f} MB")
