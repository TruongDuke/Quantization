# Báo cáo 19/10: QAT cho M-LSD và DistilBERT (INT8, TFLite)

Người làm: Trương Công Thái Đức

Ghi chú: code thí nghiệm làm cùng Claude (Claude viết và chạy script), em đọc lại kết quả và cách đo.

Hướng làm: deploy bằng TFLite/LiteRT INT8 (hướng Android), khác với hướng ONNX Runtime / GPU của Huy. Máy: Mac M1 Pro 16 GB, train bằng GPU MPS, đo tốc độ CPU 4 thread.

## 1. M-LSD tiny

**Data**
- Train: 3000 ảnh Imagenette train (không nhãn), có lật ngang và đổi độ sáng ngẫu nhiên.
- Calibration: 100 ảnh Imagenette train khác.
- Đánh giá: 21 ảnh (frame_1 + 20 ảnh Imagenette val), giống báo cáo 13/10, không trùng ảnh train.
- Metric: so với PyTorch FP32: % center tìm lại được, sai số đầu mút (pixel ảnh 512), % đoạn thẳng giữ được.

**Train bằng gì**
- PyTorch 2.13 + torchao (PT2E): `prepare_qat_pt2e` chèn fake quantize, `convert_pt2e` chuyển sang INT8.
- litert-torch (`PT2EQuantizer`, weight INT8 per-channel) để xuất TFLite INT8 thật (vào/ra INT8, chạy XNNPACK).
- Model: bản tách head center / displacement (từ báo cáo 13/10).

**Kỹ thuật training**
- Distillation: model FP32 làm thầy, không cần nhãn. Loss center = BCE với xác suất của thầy (tăng trọng số chỗ có đoạn thẳng), loss displacement = L1 có trọng số theo xác suất center của thầy (chỉ quan trọng quanh center).
- Calibration trước để khởi tạo khoảng quantize, BatchNorm đóng băng, chốt khoảng quantize sau 20% số bước, chỉ train weight.
- 150 bước, batch 4 x cộng dồn gradient 2 (= 8 ảnh/bước), Adam lr 2e-5.

**Kết quả (TFLite INT8, cùng quy trình PT2E)**

| Bản | Center | Sai số đầu mút (trung vị) | Giữ đoạn thẳng | Size | Tốc độ |
|---|---|---|---|---|---|
| FP32 | 100% | 0 | 100% | 2.56 MB | 33.1 ms |
| PTQ (chỉ calibration) | 51.6% | 4.99 px | 31.2% | 0.81 MB | 11.2 ms |
| QAT 150 bước | 61.9% | 4.18 px | 44.0% | 0.81 MB | 11.5 ms |

- QAT cải thiện rõ so với PTQ cùng quy trình (31.2% lên 44.0%), loss displacement vẫn đang giảm đều (22.2 xuống 16.1) nên train thêm còn tốt lên.
- Nhưng vẫn thấp hơn bản PTQ tốt nhất hôm 13/10 bằng ai-edge-quantizer (tách head + giữ 3 layer đầu float: 77.7%). Lý do có thể: cấu hình của PT2EQuantizer quantize activation kiểu đối xứng (phí nửa khoảng với output ReLU), quantize cả 3 layer nhạy nhất, và train còn quá ngắn.

## 1b. M-LSD: QAT bản 2, so cùng cấu hình với PTQ (chạy trên Google Colab)

Bản 1 ở trên so không công bằng: QAT dùng cấu hình mặc định của litert-torch, còn PTQ tốt nhất (77.7%) dùng ai-edge-quantizer có giữ 3 layer float. Bản 2 sửa lại:
- Fake quantize tự viết, mô phỏng đúng cấu hình của ai-edge-quantizer: weight INT8 per-channel đối xứng, activation INT8 per-tensor bất đối xứng, khoảng activation lấy bằng EMA 0.95. BatchNorm gộp vào conv trước khi train.
- Train xong gỡ fake quantize, quantize lại bằng ai-edge-quantizer giống hệt bản PTQ.
- 2 cấu hình: giữ 3 layer nhạy ở float, và quantize tất cả.
- Train 1000 bước, batch 8, Adam lr 2e-5 giảm dần (cosine), chốt khoảng activation sau 20% số bước. Data và loss giống bản 1 (distillation, 3000 ảnh Imagenette).
- Chạy trên Colab GPU T4, mỗi cấu hình khoảng 7 phút train. Torch cài thêm bị lệch phiên bản cuDNN với Colab, script tự tắt cuDNN để chạy được.

| Bản (TFLite INT8, so với FP32) | Center | Sai số đầu mút (trung vị) | Lệch dưới 2 px | Giữ đoạn thẳng |
|---|---|---|---|---|
| PTQ INT8 tất cả | 65.8% | 2.59 px | 26.9% | 58.0% |
| QAT INT8 tất cả | 77.7% | 1.96 px | 51.7% | 70.7% |
| PTQ INT8 giữ 3 float | 81.8% | 2.29 px | 35.7% | 77.6% |
| QAT INT8 giữ 3 float | 82.0% | 2.44 px | 29.6% | 83.4% |

- Cùng cấu hình thì QAT vượt PTQ ở cả 2 trường hợp: quantize tất cả 58.0% lên 70.7%, giữ 3 float 77.6% lên 83.4%.
- QAT quantize tất cả cải thiện displacement rất rõ (sai số 2.59 xuống 1.96 px, tốt nhất trong các bản). Bản này là INT8 hoàn toàn nên chạy được trên NPU/DSP.
- QAT giữ 3 float chủ yếu cải thiện center (chọn đúng điểm, ít đoạn thừa), displacement không tốt lên. Có thể do trọng số loss displacement (0.05) đang nhỏ.
- Tốc độ lấy theo số đo trên Mac (13/10): giữ 3 float khoảng 10 ms, INT8 tất cả khoảng 9 ms, FP32 33 ms. Số ms trên Colab là TFLite chạy CPU 2 nhân nên không dùng.

Code: `mlsd/qat_mlsd_v2.py`, notebook `mlsd/colab_qat_mlsd_v2.ipynb`, kết quả `mlsd/results/qat_mlsd_v2_colab.txt`.

## 2. DistilBERT (SST-2)

**Data**: GLUE SST-2. Train QAT 4000 câu, calibration 100 câu (không trùng), đánh giá 872 câu validation. Câu dài tối đa 64 token.

**Train bằng gì**: PyTorch + transformers, fake quantize tự viết (round + clamp, để chạy được trên MPS). Xuất TFLite bằng litert-torch, quantize bằng ai-edge-quantizer.

**Kỹ thuật training**
- Fake quantize cho 38 lớp Linear: weight INT8 per-channel đối xứng, input INT8 per-tensor, khoảng của input lấy bằng EMA 0.95 (giống cách ai-edge-quantizer calibration).
- LayerNorm, Softmax, GELU, attention mask giữ float.
- Loss = 0.5 x distillation (KL với model FP32, nhiệt độ 2) + 0.5 x cross-entropy với nhãn thật. 1 epoch, batch 16, AdamW lr 1e-5. Train 50 giây trên MPS.

**Kết quả (TFLite, 872 câu)**

| Bản | Accuracy | Size | Tốc độ |
|---|---|---|---|
| FP32 | 91.06% | 266.7 MB | 26.0 ms |
| PTQ dynamic | 90.83% | 68.1 MB | 8.9 ms |
| PTQ static chỉ Linear | 90.02% | 138.6 MB | 8.5 ms |
| PTQ static toàn bộ op | 66.17% | 138.4 MB | 15.8 ms |
| QAT + static chỉ Linear | 90.25% | 138.6 MB | 8.1 ms |
| QAT + static toàn bộ op | 71.22% | 138.4 MB | 15.5 ms |
| QAT + dynamic | 91.17% | 68.1 MB | 8.7 ms |

- INT8 nhanh hơn FP32 khoảng 3 lần.
- Chỉ quantize Linear thì PTQ đã gần FP32 (90.0 so với 91.1), QAT tăng thêm 0.2 điểm (2 câu), coi như không đáng kể.
- Quantize toàn bộ op thì hỏng (66%, giống kết quả 52% của Huy), QAT chỉ kéo lên 71%. Nguyên nhân là các op LayerNorm, Softmax, GELU, attention mask bị quantize INT8, mà lúc QAT em giữ float các op này nên model không học bù được cho chúng.
- Bản static chỉ Linear vẫn 138 MB vì bảng embedding (khoảng 23 triệu tham số) chưa được quantize, bản dynamic thì có nên còn 68 MB.

**Phân tích thêm (20/10): QAT đang làm những layer nào, vì sao không giúp**

- QAT chỉ fake quantize 38 lớp Linear (q, k, v, out của attention và lin1, lin2 của FFN ở 6 block, cộng pre_classifier, classifier). Các phần bản đầu giữ float (embedding, LayerNorm, Softmax, GELU, matmul trong attention, phép cộng residual) KHÔNG được fake quantize lúc QAT.
- Quantize Linear + thêm từng loại op (static INT8, 872 câu):

| Quantize | Accuracy |
|---|---|
| Chỉ Linear | 90.02% |
| + matmul trong attention | 90.60% |
| + Softmax | 90.37% |
| + GELU | 90.02% |
| + LayerNorm (MEAN, SUB, SQUARED_DIFF, RSQRT) | 88.76% |
| + ADD, MUL | 54.82% |
| Tất cả trừ ADD, MUL | 87.73% |
| Tất cả | 66.17% |

- Thủ phạm là ADD/MUL ở luồng residual và đầu ra LayerNorm: vài kênh có giá trị ngoại lai rất lớn (khoảng [-353, 4] hoặc [0.8, 349]) trong khi phần lớn giá trị chỉ vài đơn vị (khoảng trung vị của các tensor ADD/MUL là 22). INT8 per-tensor dùng 1 scale cho cả tensor nên các giá trị bình thường chỉ còn vài nấc, mất thông tin. Attention mask không phải nguyên nhân (mask làm bằng SELECT, không qua ADD). Embedding không quantize static được (ai-edge-quantizer không hỗ trợ), nên file vẫn 138 MB.
- Vì sao QAT không giúp:
  1. Với bản chỉ quantize Linear: PTQ đã 90.0% so với FP32 91.1% (lệch khoảng 9 câu), còn rất ít chỗ để cải thiện. QAT tăng 0.23 điểm (2 câu), nằm trong nhiễu đo.
  2. Train QAT gần như không đổi gì: accuracy giả lập trước và sau train đều 90.71%, loss còn tăng (cross-entropy 0.024 lên 0.079). Model gốc đã khớp nhãn và khớp FP32, 1 epoch với lr 1e-5 không học thêm được gì.
  3. Với bản quantize toàn bộ: QAT không fake quantize residual/LayerNorm, tức là không cho model "thấy" đúng chỗ gây hỏng, nên không học bù được (66% lên 71% chỉ là phụ).
- Hướng sửa: giữ ADD/MUL ở float (87.7%, gần INT8 toàn bộ), hoặc QAT có fake quantize cả residual/LayerNorm, hoặc dùng activation 16-bit / SmoothQuant cho các tensor có ngoại lai.

## 3. Khó khăn gặp phải

- litert-torch và torchao bản hiện tại không tương thích ở phần QAT conv + BatchNorm (litert-torch truyền hàm, torchao chỉ nhận nn.Module). Đã vá lúc chạy.
- Export phải để chiều batch động (train batch > 1, convert batch 1).
- Fake quantize của torchao chưa chạy trên MPS, phải chép qua CPU mỗi layer, nên train chậm. Máy 16 GB còn chạy app khác, batch 8 ảnh 512x512 tốn 7.7 GB làm máy swap, phải giảm batch và cộng dồn gradient. 150 bước mất khoảng 2 giờ (lúc không bị swap chỉ khoảng 3 giây/bước).
- Với DistilBERT, em tự viết fake quantize bằng phép tính thường để chạy được trên MPS (50 giây cho 1 epoch 4000 câu).

## 4. Kết luận

- M-LSD: khi so cùng cấu hình, QAT vượt PTQ: giữ 3 float 77.6% lên 83.4%, quantize tất cả 58.0% lên 70.7% (sai số đầu mút còn 1.96 px).
- DistilBERT: chỉ quantize Linear thì PTQ đã đủ tốt, QAT gần như không thêm gì. Quantize toàn bộ op thì phải giữ LayerNorm / Softmax / GELU ở float, QAT không cứu được.
- Với DistilBERT thì giống kết luận của Huy (chọn đúng phần để quantize là đủ). Với M-LSD thì QAT có ích thật, nhất là khi muốn INT8 hoàn toàn.

## 5. Tiếp theo

- M-LSD: tăng trọng số loss displacement cho bản giữ 3 float, train lâu hơn, đo trên điện thoại.
- DistilBERT: quantize cả embedding để giảm kích thước, thử QAT có fake quantize cả LayerNorm/Softmax nếu muốn quantize toàn bộ.
- Đo trên điện thoại Android thật.

Code: `mlsd/qat_mlsd.py`, `distilbert/qat_distilbert.py`. Kết quả: `mlsd/results/`, `distilbert/results/`.
