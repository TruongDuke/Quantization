# Quantization

Luyện tập về deploy model và quantization.

| Thư mục | Nội dung |
|---|---|
| `0-numpy/` | Tự cài quantize/dequantize bằng numpy: round-trip, clip và round, chọn khoảng, per-tensor và per-channel |
| `2.1/` | Ví dụ phép nhân ma trận bằng số nguyên |
| `3-deploy/` | Convert MobileNetV2 từ PyTorch sang ONNX và TFLite (FP32), so sai khác output (SNR) và tốc độ, chạy trên giả lập Android |
| `4-quantization/` | Quantize INT8 ResNet18 bằng ONNX Runtime và TFLite (dynamic, static), đánh giá accuracy, SNR, tốc độ. Báo cáo: `4-quantization/BAO_CAO.md` |

Dữ liệu, file model và file `.npy` không đưa lên repo, chạy lại script để tạo. Hướng dẫn chạy: `4-quantization/README.md`.
