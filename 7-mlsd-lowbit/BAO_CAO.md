# Báo cáo 13/10: M-LSD ở bit thấp (post-training quantization)

Người làm: Trương Công Thái Đức

Ghi chú: code thí nghiệm hôm nay em làm cùng Claude (Claude viết và chạy script, em đọc lại kết quả và cách đo).

Model: M-LSD tiny 512 (bản PyTorch lhwcv), convert bằng litert-torch, quantize bằng ai-edge-quantizer. Đánh giá trên 21 ảnh (frame_1 + 20 ảnh Imagenette val), calibration 100 ảnh Imagenette train. Mọi so sánh là so với model PyTorch FP32. Máy Mac M1 Pro, CPU 4 thread.

Số liệu đầy đủ: `results/summary.txt` (và các file json trong `results/`).

## 1. Sửa cách đo: tách phần classification và regression

M-LSD là model hybrid:
- Center map (classification): mỗi pixel có phải tâm đoạn thẳng không. Chỉ cần thứ tự điểm cao/thấp đúng.
- Displacement (regression): độ dời tới 2 đầu mút, khoảng ±131 pixel. Cần chính xác về giá trị.

Em đọc loss lúc train (`mlsd_pytorch/loss/mlsd_multi_loss.py`) thì thấy 2 điều:
- Displacement chỉ được train tại các pixel nằm trên đoạn thẳng thật. Các pixel còn lại không bị ràng buộc gì.
- Loss lấy min giữa 2 thứ tự đầu mút (start/end hay end/start đều đúng).

Nên tính SNR trên toàn bộ bản đồ (như báo cáo trước) bị các vùng "rác" kéo xuống. Metric mới:
- Center: % center của FP32 vẫn tìm lại được (lệch không quá 1 pixel).
- Displacement: sai số đầu mút tính tại đúng các center được dùng, không phân biệt thứ tự start/end (pixel trên ảnh 512).
- Đoạn thẳng: % đoạn của FP32 vẫn tìm lại được (2 đầu mút lệch dưới 5 pixel).

## 2. FP16 thật ra không tệ

| Bản | SNR displacement cũ | Center | Sai số đầu mút (trung vị) | Giữ đoạn thẳng |
|---|---|---|---|---|
| FP16 weight (TFLite) | 21.0 dB | 97.9% | 0.07 px | 97.3% |
| FP16 toàn bộ (PyTorch) | 19.8 dB | 97.8% | 0.09 px | 97.4% |

SNR toàn bản đồ chỉ 21 dB nhưng tại các điểm thật sự được dùng thì đầu mút chỉ lệch khoảng 0.07 pixel, giữ được 97% đoạn thẳng. Vậy FP16 dùng được, con số SNR thấp là do cách đo.

## 3. INT8: điểm xuất phát

| Bản | Center | Sai số đầu mút | Giữ đoạn thẳng | Tốc độ |
|---|---|---|---|---|
| FP32 | 100% | 0 | 100% | 33.1 ms |
| INT8 weight-only | 72.6% | 1.19 px | 62.5% | 33.2 ms |
| INT8 dynamic | 71.5% | 1.22 px | 59.6% | 20.5 ms |
| INT8 static | 27.2% | 2.59 px | 37.1% | 9.1 ms |

Chỉ quantize weight (activation vẫn float) đã mất gần 40% đoạn thẳng, nên không chỉ activation mà weight của một số layer cũng có vấn đề.

## 4. Sửa phần output (static INT8)

| Thay đổi (mỗi lần 1 thứ) | Center | Giữ đoạn thẳng |
|---|---|---|
| Gốc: 9 kênh chung 1 tensor | 27.2% | 37.1% |
| Bỏ 4 kênh không dùng (còn 5 kênh) | 27.2% | 37.1% |
| 5 kênh + chia displacement cho 8 | 60.9% | 60.5% |
| Tách center và displacement thành 2 tensor | 65.8% | 57.7% |

- Các thay đổi này không làm đổi kết quả FP32 (đã kiểm tra lệch dưới 1e-4).
- Chuẩn hoá hoặc tách head làm center tốt lên rõ (27% lên 61-66%), vì center không còn bị kênh displacement (±131) chiếm hết scale.
- Sai số đầu mút gần như không đổi (khoảng 2.6 px), phần này cần cách khác.

## 5. Tìm layer nhạy (trên bản tách head)

Mỗi lần chỉ quantize 1 layer, các layer khác để float:

| Layer | Giữ đoạn thẳng |
|---|---|
| Depthwise conv đầu tiên (InvertedResidual_1) | 65.1% |
| Conv 1x1 sau nó (InvertedResidual_1) | 87.2% |
| Conv đầu tiên của backbone | 87.7% |
| Các conv ở phần head (block16, head center) | 93-96% |
| Các layer còn lại | 95-99% |

Riêng layer depthwise đầu tiên đã làm mất 35% đoạn thẳng. Đây là các layer làm việc ở độ phân giải cao nhất (256x256) ngay sau input.

## 6. Mixed precision: giữ vài layer nhạy ở float

| Giữ float | Center | Giữ đoạn thẳng | Size | Tốc độ |
|---|---|---|---|---|
| 0 layer (INT8 hết) | 65.8% | 57.7% | 0.86 MB | 9.2 ms |
| 3 layer nhạy nhất | 81.9% | 77.7% | 0.87 MB | 10.1 ms |
| 8 layer nhạy nhất | 82.8% | 84.6% | 1.10 MB | 17.4 ms |
| 3 layer + head displacement | 81.9% | 77.7% | 0.87 MB | 10.0 ms |

Bản tốt nhất về cân bằng: tách head + 3 layer đầu giữ float. Từ 37.1% (INT8 gốc) lên 77.7%, vẫn nhanh gần bằng INT8 (10.1 ms so với 33.1 ms của FP32), file gần như không to thêm.

Giữ head displacement ở float không giúp gì, nên sai số đầu mút còn lại (khoảng 2.3 px) đến từ sai số activation tích luỹ qua các layer trước, không phải từ weight của head.

## 7. Cách chọn khoảng khi calibration

| Cách | Giữ đoạn thẳng (bản tách head) |
|---|---|
| MinMax + EMA (mặc định) | 57.7% |
| MinMax toàn cục | 26.2% |
| MSE | 26.5% |
| OCTAV | 55.3% |

Với model này, MinMax toàn cục tệ hơn EMA (ngược với ResNet18): ảnh calibration có giá trị ngoại lai lớn làm khoảng quá rộng, displacement bị làm tròn thô (sai số đầu mút 6 px). MSE và OCTAV không tốt hơn mặc định.

## 8. Kết luận

- FP16 dùng được (giữ 97% đoạn thẳng). Đánh giá "FP16 sai số nhiều" trước đó là do SNR tính trên toàn bản đồ.
- INT8 tệ vì 2 nguyên nhân: center và displacement dùng chung scale ở output, và vài layer đầu backbone rất nhạy.
- Sửa bằng PTQ (tách head + giữ 3 layer đầu float): 37.1% lên 77.7% đoạn thẳng, tốc độ gần bằng INT8.
- Còn lại: displacement lệch khoảng 2.3 px do activation INT8 tích luỹ, PTQ khó sửa thêm.

## 9. Hạn chế và tiếp theo

Hạn chế:
- Chỉ so với FP32, chưa có nhãn thật (chưa đo sAP).
- Ảnh calibration/đánh giá là Imagenette, không phải ảnh nhiều đường thẳng.
- 21 ảnh đánh giá còn ít.

Tiếp theo:
- QAT: repo PyTorch có sẵn code train, fine-tune vài epoch có mô phỏng quantize để model tự học bù sai số displacement.
- Tải bộ Wireframe: calibration bằng ảnh đúng loại và đo sAP10 như paper.
- Thử activation INT16 chỉ cho các layer cuối (không phải cả model).
