import torch, torchvision
import numpy as np

#load model
model = torchvision.models.mobilenet_v2(weights="DEFAULT")
model.eval()

#generate random input
x = torch.rand(1, 3, 224, 224)

with torch.no_grad(): # Disable gradient computation, gardien for training
    y = model(x)

print("Output shape:", y.shape)
print("Model doan class so:", y.argmax().item())

np.save("input.npy", x.numpy())
np.save("output.npy", y.numpy())
print("Input and output saved to input.npy and output.npy")
