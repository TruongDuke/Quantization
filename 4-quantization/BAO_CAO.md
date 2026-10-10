# Báo cáo quantization - 08/10

Người làm: Trương Công Thái Đức

## 1. Hôm nay làm gì

- Tìm hiểu quantization INT8 trên ONNX Runtime và TFLite
- Thử với ResNet18 (hôm qua làm MobileNetV2)
- Quantize theo 2 cách dynamic và static, rồi so với model gốc

## 2. Tìm hiểu

Quantization là đổi số FP32 (4 byte) sang INT8 (1 byte) theo công thức:

    q = round(x / scale) + zero_point

Model nhỏ đi khoảng 4 lần và chạy nhanh hơn, nhưng bị sai số do làm tròn.

Có 2 cách chính:
- Dynamic: chỉ quantize weight trước. Activation thì lúc chạy mới tính scale, theo từng ảnh. Không cần dữ liệu calibration, nhưng chạy chậm hơn static. Hợp với Transformer, LSTM hơn là CNN.
- Static: quantize cả weight và activation. Phải cho model chạy qua một ít ảnh thật trước (calibration) để đo min/max của activation. Chạy nhanh nhất, và chạy được trên NPU của điện thoại.

Công cụ dùng:
- ONNX Runtime: `quantize_dynamic`, `quantize_static`
- TFLite: thư viện `ai-edge-quantizer`, dùng 2 recipe là `dynamic_wi8_afp32` và `static_wi8_ai8`

## 3. Cách làm

- Model: ResNet18 pretrained ImageNet
- Dữ liệu: bộ Imagenette (10 class của ImageNet). Lấy 100 ảnh train để calibration, 300 ảnh val để đánh giá.
- Convert PyTorch sang ONNX và TFLite (FP32), sau đó quantize mỗi bên theo 2 cách
- Đo trên máy Mac M1 Pro, CPU 4 thread

## 4. Kết quả

Model gốc PyTorch đúng 64.7%.

| Model | Size (MB) | Acc (%) | Giống model gốc (%) | SNR (dB) | Thời gian (ms) |
|---|---|---|---|---|---|
| ONNX FP32 | 46.8 | 64.7 | 100 | 110.3 | 14.9 |
| ONNX dynamic | 11.8 | 65.7 | 93.3 | 20.5 | 9.5 |
| ONNX static | 11.9 | 68.0 | 96.3 | 24.6 | 3.7 |
| TFLite FP32 | 46.8 | 64.7 | 100 | 109.9 | 15.3 |
| TFLite dynamic | 11.8 | 67.3 | 95.3 | 27.7 | 4.7 |
| TFLite static | 11.9 | 61.0 | 85.0 | 20.9 | 4.3 |

"Giống model gốc" là số % ảnh mà model đoán cùng class với PyTorch gốc.

## 5. Nhận xét

- Quantize xong model nhỏ đi 4 lần (47MB còn 12MB).
- SNR giảm từ khoảng 110 dB xuống còn 20–28 dB, nhưng model vẫn đoán giống bản gốc 85–96%.
- Static nhanh nhất, ONNX static nhanh gấp 4 lần FP32. ONNX dynamic chỉ nhanh hơn khoảng 1.6 lần.
- Có vài bản INT8 accuracy cao hơn model gốc. Đây chỉ là ngẫu nhiên, vì chỉ có 300 ảnh và chênh nhau khoảng 10 ảnh. Nên nhìn cột "giống model gốc" thì đáng tin hơn.
- TFLite static kém nhất (85%). Em kiểm tra thì thấy output bị kẹp. Lúc calibration nó ước lượng giá trị lớn nhất của output là khoảng 14.6, nhưng thực tế lên tới 32.5. Các giá trị lớn bị kẹp hết về cùng một mức, nên có 62/300 ảnh bị hoà điểm giữa các class và đoán sai. (Nguyên nhân vì sao ước lượng thiếu: xem `BAO_CAO_09_10.md`.)

## 6. Kết luận

- Chạy trên máy tính thì nên dùng ONNX static: nhanh nhất mà vẫn giữ được 96% kết quả.
- Chạy trên Android thì tạm thời dùng TFLite dynamic. Sau khi sửa được lỗi kẹp output thì nên chuyển sang TFLite static.

## 7. Lỗi gặp phải

- File ONNX bị tách thành 2 file (`.onnx` và `.onnx.data`), phải gộp lại trước khi quantize.
- onnxruntime bản mới đã bỏ tham số `optimize_model`.
- Model TFLite static nhận input INT8 chứ không phải float, phải tự đổi input và output.

## 8. Tiếp theo

- Sửa lỗi kẹp output của TFLite static (giữ layer cuối là FP32, hoặc calibration nhiều ảnh hơn)
- Thử quantize MobileNetV2 để so sánh
- Đo trên điện thoại thật
