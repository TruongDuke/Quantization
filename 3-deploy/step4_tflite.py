import numpy as np
import torch
import torchvision
import litert_torch as litorch
from ai_edge_litert.interpreter import Interpreter

#convert pytorch model to tflite
model = torchvision.models.mobilenet_v2(weights="DEFAULT")
model.eval()
x = np.load("input.npy")
ref = np.load("output.npy")

edge_model = litorch.convert(model, (torch.from_numpy(x),))
edge_model.export("mobilenet_v2.tflite")
print("Model exported to mobilenet_v2.tflite")

#load tflite model and run inference
interpreter = Interpreter(model_path="mobilenet_v2.tflite", num_threads=4)
interpreter.allocate_tensors()

input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]
print("Input shape:", input_info["shape"], "| Output shape:", output_info["shape"])
interpreter.set_tensor(input_info["index"], x)
interpreter.invoke()
out = interpreter.get_tensor(output_info["index"])

print("Model doan class so:", out.argmax())
print("Sai so lon nhat:", np.max(np.abs(out - ref)))