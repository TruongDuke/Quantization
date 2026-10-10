# Báo cáo 09/10

Người làm: Trương Công Thái Đức

Ghi chú: code thí nghiệm có viết cùng Claude. Nguyên nhân EMA em tìm ra khi đọc source của ai-edge-quantizer cùng Claude, sau đó em tự kiểm chứng bằng thí nghiệm (q5, q6).

## 1. Vì sao TFLite calibration ước lượng sai khoảng output

Nhắc lại: hôm 08/10, ResNet18 TFLite static chỉ giữ được 85% dự đoán giống model gốc, ONNX static được 96.3%. Lý do trực tiếp là output TFLite bị kẹp ở 14.6 trong khi logit thật lên tới 32.5.

### Nguyên nhân

Hai thư viện gộp min/max của các ảnh calibration khác nhau:
- ONNX Runtime: lấy min/max toàn cục, tức là giá trị lớn nhất từng gặp trong tất cả ảnh (`onnxruntime/quantization/calibrate.py`, mặc định `moving_average=False`).
- TFLite (ai-edge-quantizer): lấy trung bình trượt EMA, `max_mới = 0.95 * max_cũ + 0.05 * max_ảnh_mới` (`ai_edge_quantizer/utils/qsv_utils.py`, hàm `moving_average_update`).

EMA cho ra kiểu "max trung bình của từng ảnh" (khoảng 15), chứ không phải max lớn nhất (32.5). Ảnh nào có logit rất cao chỉ góp 5%, nên khoảng bị ước lượng thiếu và output bị kẹp.

Ngoài ra EMA phụ thuộc thứ tự ảnh: ảnh càng về sau càng có trọng số lớn. Bộ calibration của em xếp theo class nên kết quả gần như chỉ phản ánh mấy class cuối.

### Thí nghiệm (q5_calib_subsets.py)

Cùng một bộ ảnh calibration thì quantize cả ONNX và TFLite, rồi đo khoảng output và % dự đoán giống model gốc (agree) trên 300 ảnh val.

**B. Cùng 100 ảnh, chỉ đổi thứ tự**

| Thứ tự | ONNX | TFLite |
|---|---|---|
| Theo class | [-9.1, 31.5], 96.3% | [-6.8, 14.6], 85.0% |
| Đảo ngược | [-9.1, 31.5], 96.3% | [-6.9, 15.7], 90.3% |
| Trộn | [-9.1, 31.5], 96.3% | [-6.7, 14.3], 86.7% |

ONNX giống hệt nhau, TFLite thay đổi theo thứ tự dù dữ liệu y hệt.

**A. 5 subset ngẫu nhiên (100 ảnh mỗi subset, lấy từ 500 ảnh train)**

| Seed | ONNX | TFLite |
|---|---|---|
| 0 | [-9.5, 28.4], 96.0% | [-6.9, 15.7], 90.7% |
| 1 | [-9.9, 27.9], 95.0% | [-6.8, 16.3], 90.0% |
| 2 | [-9.9, 34.6], 94.7% | [-7.0, 15.9], 90.0% |
| 3 | [-9.1, 34.6], 93.3% | [-6.6, 16.4], 91.3% |
| 4 | [-9.2, 28.2], 96.0% | [-6.7, 15.6], 90.0% |

Max của TFLite luôn khoảng 16, không bao giờ tới 32.5. Max của ONNX thay đổi tuỳ subset có ảnh "cực đoan" hay không.

**C. Số lượng ảnh calibration**

| Số ảnh | Max ONNX | Max TFLite |
|---|---|---|
| 10 | 17.7 | 14.8 |
| 50 | 26.1 | 14.5 |
| 100 | 34.5 | 16.6 |
| 300 | 34.6 | 15.2 |

ONNX càng nhiều ảnh thì max càng tăng (càng dễ gặp giá trị cực đại). Với 10 ảnh ONNX cũng bị thiếu. TFLite thêm ảnh cũng không thay đổi vì EMA hội tụ về max trung bình.

Thêm một điểm: khoảng rộng hơn chưa chắc tốt hơn. Seed 2 và 3 của ONNX có max cao nhất nhưng agree thấp nhất, vì khoảng rộng thì mỗi nấc INT8 to hơn, sai số làm tròn tăng (đánh đổi giữa clipping và rounding).

### Kiểm chứng nguyên nhân (q6_fix_tflite_calib.py)

Em thay hàm EMA của TFLite bằng min/max toàn cục (monkey patch, không sửa file thư viện), giữ nguyên mọi thứ khác:

| Cách tính | Thứ tự | Khoảng output | Agree | SNR |
|---|---|---|---|---|
| EMA (mặc định) | theo class | [-6.8, 14.6] | 85.0% | 20.9 dB |
| EMA (mặc định) | đảo ngược | [-6.9, 15.7] | 90.3% | 21.9 dB |
| Min/max toàn cục | theo class | [-9.1, 31.5] | 95.7% | 24.6 dB |
| Min/max toàn cục | đảo ngược | [-9.1, 31.5] | 95.7% | 24.6 dB |

Chỉ đổi cách tính min/max mà TFLite lên 95.7%, khoảng output giống hệt ONNX và không còn phụ thuộc thứ tự. Vậy nguyên nhân là EMA.

## 2. Dynamic/static của ONNX Runtime khác TFLite thế nào

Cùng tên gọi nhưng cấu hình mặc định khác nhau:

**Dynamic**

| | ONNX Runtime | TFLite |
|---|---|---|
| Weight của Conv | per-tensor (1 scale cho cả layer) | per-channel (mỗi kênh 1 scale) |
| Cách chạy | tách thành nhiều node: DynamicQuantizeLinear, ConvInteger, nhân lại scale | 1 kernel hybrid của XNNPACK |
| Kết quả ResNet18 | SNR 20.5 dB, 9.45 ms | SNR 27.7 dB, 4.67 ms |

TFLite nhanh gấp đôi vì kernel gộp, và chính xác hơn nhiều khả năng do per-channel.

Em thử bật `per_channel=True` cho ONNX dynamic nhưng SNR không đổi (20.5 lên 20.6). Kiểm tra file model thì scale của weight vẫn chỉ có 1 số mỗi layer: ONNX Runtime bỏ qua cờ này với Conv khi quantize dynamic (trong `onnxruntime/quantization/operators/conv.py`, ConvInteger gọi quantize_weight không có per-channel), và không báo lỗi gì. Vì vậy em chưa tách riêng được ảnh hưởng của per-channel, chỉ nói được là nhiều khả năng.

Bài học: đặt tham số xong phải kiểm tra file đầu ra xem có thực sự được áp dụng không.

**Static**

| | ONNX Runtime | TFLite |
|---|---|---|
| Weight | per-channel | per-channel |
| Activation | per-tensor | per-tensor |
| Gộp min/max khi calibration | toàn cục | EMA |
| Input/output model | float (quantize nằm trong đồ thị) | INT8 |

Cấu hình gần giống nhau, khác biệt lớn nhất là cách gộp min/max (mục 1). Tốc độ gần bằng nhau (3.68 và 4.27 ms).

Kết luận: so sánh 2 framework thì phải đặt cấu hình giống nhau, không nên dùng mặc định.

## 3. Hướng tiếp theo đã làm

Sửa TFLite static bằng min/max toàn cục (q6): từ 85% lên 95.7%.
