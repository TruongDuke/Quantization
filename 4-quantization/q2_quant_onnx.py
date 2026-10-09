import os
import numpy as np
from onnxruntime.quantization import (
    quantize_dynamic, quantize_static,
    CalibrationDataReader, QuantType, QuantFormat,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

# tien xy ly model
quant_pre_process("resnet18_fp32.onnx","resnet18_prep.onnx")

# Dynamic quantization
quantize_dynamic("resnet18_prep.onnx", "resnet18_onnx_dynamic.onnx", weight_type=QuantType.QInt8, per_channel=True)
print("Dynamic quantization done.")

# Static quantization
calib_x = np.load("calib_x.npy")

class MyCalibReader(CalibrationDataReader):
    def __init__(self, data):
        self.data = iter([{"input": data[i:i+1]} for i in range(len(data))])

    def get_next(self):
        return next(self.data, None)

quantize_static(
    "resnet18_prep.onnx",
    "resnet18_onnx_static.onnx",
    MyCalibReader(calib_x),
    quant_format=QuantFormat.QDQ,
    per_channel=True,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
)
print("Static Done")

# so sanh kich thuoc
for f in ["resnet18_fp32.onnx", "resnet18_onnx_dynamic.onnx", "resnet18_onnx_static.onnx"]:
    print(f"{f:30s} {os.path.getsize(f) / 1e6:.1f} MB")