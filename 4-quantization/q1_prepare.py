import glob
import numpy as np
import torch
import torchvision
import onnx
import litert_torch
from PIL import Image
from torchvision.models import ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

CLASSES = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491, "n03028079": 497,
           "n03394916": 566, "n03417042": 569, "n03425413": 571, "n03445777": 574, "n03888257": 701}

# chuan bi anh that 
def load_images(split, n_per_class):
    images, labels = [], []
    for folder, class_id in CLASSES.items():
        paths = sorted(glob.glob(f"imagenette2-160/{split}/{folder}/*.JPEG"))[:n_per_class]
        for p in paths:
            img = Image.open(p).convert("RGB")
            x = preprocess(img).numpy()
            images.append(x)
            labels.append(class_id)
    return np.stack(images).astype(np.float32), np.array(labels)

calib_x, _ = load_images("train", 10)
val_x, val_y = load_images("val", 30)
np.save("calib_x.npy", calib_x)
np.save("val_x.npy", val_x)
np.save("val_y.npy", val_y)
print("Calib:", calib_x.shape, "| Val:", val_x.shape, val_y.shape)

# chuan bi model
model = torchvision.models.resnet18(weights=weights).eval()
sample = torch.from_numpy(calib_x[:1])

torch.onnx.export(model, sample, "resnet18.onnx", input_names=["input"], output_names=["output"],
                  opset_version=18, dynamo= True)
onnx.save(onnx.load("resnet18.onnx"), "resnet18_fp32.onnx")
print("Saved resnet18_fp32.onnx")

# chuan bi model cho quantization
onnx.save(onnx.load("resnet18_fp32.onnx"), "resnet18_quant.onnx")

litert_torch.convert(model, (sample,)).export("resnet18_fp32.tflite")

# result
with torch.no_grad():
    ref = model(torch.from_numpy(val_x)).numpy()
np.save("ref_output.npy", ref)
print("PyTorch accuracy:", np.mean(ref.argmax(1) == val_y) * 100, "%")
