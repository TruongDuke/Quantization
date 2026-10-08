import torch
import torchvision
import numpy as np
import onnxruntime as ort

# Export pytorch model to ONNX
model = torchvision.models.mobilenet_v2(weights="DEFAULT")
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy,
    "mobilenet_v2.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    dynamo=True,

)
print("Model exported to mobilenet_v2.onnx")

# Load ONNX model and run inference
x = np.load("input.npy")
ref = np.load("output.npy")

session = ort.InferenceSession("mobilenet_v2.onnx",providers=["CPUExecutionProvider"])
out = session.run(["output"], {"input": x})[0]

print("Output shape:", out.shape)
print("Model doan class so:", out.argmax().item())
print("Sai so lon nhat:", np.max(np.abs(out - ref)))