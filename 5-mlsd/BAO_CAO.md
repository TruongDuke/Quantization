# Báo cáo 10/10

Người làm: Trương Công Thái Đức

Ghi chú: code thí nghiệm có viết cùng Claude, phần giải thích thuật toán em có dùng Claude để đọc hiểu rồi tự viết lại theo cách hiểu của em.

## 1. Chạy lại code của Huy

Dựng môi trường của Huy trên máy em (Mac M1 Pro): TF 2.15.1, ORT 1.20.1, numpy 1.26 (bỏ 3 gói chỉ dành cho Windows).

Code của Huy chạy được hết trên Mac. Accuracy, SNR gần như giống báo cáo của Huy. Tốc độ (1 thread):

| Biến thể | Windows i7 của Huy (ms) | Mac M1 Pro của em (ms) |
|---|---|---|
| ONNX FP32 | 80.9 | 105.0 |
| ONNX static | 57.0 | 67.1 |
| TFLite FP32 | 409.0 | 113.0 |
| TFLite dynamic | 1007.0 | 31.0 |
| TFLite static | 326.5 | 24.3 |

Nhận xét:
- ONNX chạy trên máy Huy nhanh hơn khoảng 20-30%.
- TFLite chạy trên Mac nhanh hơn rất nhiều (dynamic nhanh hơn khoảng 32 lần). Em nghĩ là do TF 2.15 bản Windows dùng kernel chậm, còn trên Mac ARM có kernel tối ưu (XNNPACK). Cái này chưa kiểm chứng.

## 2. Chạy model của em trên môi trường của Huy

Cùng máy Mac, cùng model ResNet18, chỉ đổi thư viện:

| Model | Agree (thư viện em / Huy) | 1 thread ms (em / Huy) |
|---|---|---|
| ONNX FP32 | 100% / 100% | 47.5 / 48.3 |
| ONNX dynamic | 93.0% / lỗi | 17.7 / - |
| ONNX static | 96.3% / 96.3% | 11.0 / 30.8 |
| TFLite FP32 | 100% / 100% | 53.2 / 52.3 |
| TFLite dynamic | 95.3% / 0.0% | 11.5 / 13.4 |
| TFLite static | 85.0% / 85.3% | 11.6 / 38.8 |

Nhận xét:
- FP32 giống nhau ở 2 môi trường.
- INT8 static chạy trên thư viện mới nhanh hơn khoảng 3 lần.
- ONNX dynamic không chạy được trên ORT 1.20 (báo lỗi không có kernel ConvInteger).
- TFLite dynamic chạy được trên TF 2.15 nhưng kết quả sai hết (0%), không báo lỗi gì. Model tạo bằng công cụ mới thì runtime cũ có thể đọc sai.
- Bài học: model phải được kiểm tra cả độ chính xác trên đúng runtime sẽ dùng, không chỉ đo tốc độ.

## 3. M-LSD

### 3.1. Chạy thử
Chạy được cả bản PyTorch (lhwcv) và bản TFLite chính thức (NAVER) trên ảnh mẫu, cả 2 đều detect được cạnh quyển sách và mép ghế.

Kiến trúc bản tiny: backbone MobileNetV2 bị cắt ngắn (chỉ giữ 4 nhóm block đầu), input 4 kênh (RGB + 1 kênh toàn số 1), output (1, 9, 256, 256). Chỉ có khoảng 620 nghìn tham số nhưng cần 4.2 G MACs (nặng hơn ResNet18) vì xử lý ảnh 512x512.

### 3.2. Convert PyTorch sang TFLite
- Bản em convert khớp với PyTorch: SNR 96.2 dB.
- So với bản TFLite chính thức thì output lệch nhiều (SNR 7.5 dB, tương quan khoảng 0.9). Em kiểm tra thì không phải do cách chuẩn hoá input hay thứ tự kênh, mà nhiều khả năng do weight của repo PyTorch là weight tự train lại (README ghi sAP 56.4, paper là 58.0).
- Tốc độ bằng nhau:

| Model | 1 thread (ms) | 4 thread (ms) |
|---|---|---|
| Em convert, chỉ model | 120.8 | 33.4 |
| Em convert, có decoder | 121.4 | 33.9 |
| Chính thức FP32 | 122.0 | 34.8 |
| Chính thức FP16 | 122.3 | 34.8 |

FP16 không nhanh hơn FP32 trên CPU, chỉ giúp file nhỏ hơn.

### 3.3. Quantize
Calibration bằng 100 ảnh Imagenette (không phải loại ảnh có nhiều đường thẳng, đây là hạn chế). Đánh giá trên 21 ảnh, so với bản FP32.

| Model | Size (MB) | Giữ được đoạn thẳng | 4 thread (ms) |
|---|---|---|---|
| FP32 | 2.57 | 100% | 33.7 |
| Dynamic | 0.81 | 59.6% | 21.6 |
| Static (EMA) | 0.88 | 37.1% | 9.4 |
| Static (min/max toàn cục) | 0.88 | 8.9% | 9.4 |
| Static activation INT16 | 0.90 | 32.7% | 690.6 |

Nhận xét:
- Quantize nhanh hơn khoảng 3.6 lần nhưng mất nhiều đoạn thẳng, M-LSD khó quantize hơn ResNet18 nhiều.
- Lý do chính: 9 kênh output dùng chung 1 scale. Kênh displacement có giá trị tới ±131 nên kênh center (chỉ khoảng -17 đến 3) bị làm tròn còn vài nấc, vị trí đoạn thẳng bị sai.
- Ở bài này min/max toàn cục lại tệ hơn EMA (ngược với ResNet18), vì vài ảnh calibration có giá trị ngoại lai rất lớn. Không có cách tính min/max nào luôn đúng.
- INT16 chậm hơn 20 lần vì XNNPACK không chạy INT16, và cũng không chính xác hơn. Nguyên nhân độ chính xác em chưa tìm ra.

### 3.4. Post-processing và port Android
Em viết giải thích thuật toán và cách port trong file `ALGORITHM.md`.

## 4. Tiếp theo
- Thử giữ các layer cuối của M-LSD ở FP32 khi quantize.
- Calibration M-LSD bằng ảnh phòng ốc / tài liệu.
- Viết thử pred_lines bằng Java và kiểm tra với kết quả Python.
