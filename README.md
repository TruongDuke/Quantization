# Quantization

Luyện tập về deploy model và quantization.

| Thư mục | Nội dung |
|---|---|
| `0-numpy/` | Tự cài quantize/dequantize bằng numpy: round-trip, clip và round, chọn khoảng, per-tensor và per-channel |
| `2.1/` | Ví dụ phép nhân ma trận bằng số nguyên |
| `3-deploy/` | Convert MobileNetV2 từ PyTorch sang ONNX và TFLite (FP32), so sai khác output (SNR) và tốc độ, chạy trên giả lập Android |
| `4-quantization/` | Quantize INT8 ResNet18 bằng ONNX Runtime và TFLite (dynamic, static), đánh giá accuracy, SNR, tốc độ. Báo cáo: `4-quantization/BAO_CAO.md`. Thêm: thí nghiệm calibration TFLite/ONNX (q5, q6), so môi trường với repo của Huy (q7) |
| `9-zipformer-litert/` | Convert Zipformer streaming (sherpa-onnx, 20M) từ ONNX sang LiteRT 2.x bằng onnx2tf: các lỗi khi convert và cách sửa graph |
| `8-qat/` | QAT cho M-LSD (PT2E, distillation) và DistilBERT (SST-2), xuất TFLite INT8. Báo cáo: `8-qat/BAO_CAO.md` |
| `7-mlsd-lowbit/` | M-LSD ở bit thấp: metric tách center/displacement, FP16, tách head, độ nhạy từng layer, mixed precision. Báo cáo: `7-mlsd-lowbit/BAO_CAO.md` |
| `5-mlsd/` | M-LSD: convert PyTorch sang TFLite, so với bản chính thức, quantize, giải thích post-processing và cách port Android. Báo cáo: `5-mlsd/BAO_CAO.md`, thuật toán: `5-mlsd/ALGORITHM.md` |

Dữ liệu, file model và file `.npy` không đưa lên repo, chạy lại script để tạo. Hướng dẫn chạy: `4-quantization/README.md`.
