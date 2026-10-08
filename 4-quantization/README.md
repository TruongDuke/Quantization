# Quantization ResNet18 với ONNX Runtime và TFLite

Báo cáo kết quả: xem `BAO_CAO.md`.

## Cài đặt

Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tải dữ liệu (Imagenette, khoảng 100MB)

```bash
curl -L https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz | tar xz
```

## Chạy theo thứ tự

| File | Việc làm | Tạo ra |
|---|---|---|
| `q1_prepare.py` | Đọc ảnh (100 calibration, 300 đánh giá), export ResNet18 sang ONNX và TFLite (FP32), lưu output PyTorch gốc | `calib_x.npy`, `val_x.npy`, `val_y.npy`, `ref_output.npy`, `resnet18_fp32.onnx`, `resnet18_fp32.tflite` |
| `q2_quant_onnx.py` | Quantize ONNX: dynamic và static INT8 | `resnet18_onnx_dynamic.onnx`, `resnet18_onnx_static.onnx` |
| `q3_quant_tflite.py` | Quantize TFLite: dynamic và static INT8 | `resnet18_tflite_dynamic.tflite`, `resnet18_tflite_static.tflite` |
| `q4_evaluate.py` | Đánh giá 6 model: kích thước, accuracy, agreement, SNR, tốc độ | Bảng kết quả in ra màn hình |

```bash
python q1_prepare.py
python q2_quant_onnx.py
python q3_quant_tflite.py
python q4_evaluate.py
```
