# Thuật toán post-processing của M-LSD

File gốc: `mlsd-official/utils.py` (hàm `pred_lines` và `pred_squares`).

## pred_lines

Model không trả ra luôn toạ độ đoạn thẳng, mà trả ra 3 thứ:
- các ứng viên làm center của đoạn thẳng (200 điểm),
- độ tin cậy của từng center,
- bản đồ độ dời: tại mỗi pixel, 2 đầu mút cách pixel đó bao xa.

Các bước:
1. Với mỗi center, tính 2 đầu mút = toạ độ center + độ dời đọc tại vị trí center đó.
2. Bỏ những đoạn không đủ tin cậy (score < score_thr) và những đoạn quá ngắn (độ dài < dist_thr).
3. Nhân 2 vì output model (256x256) nhỏ bằng nửa input (512x512), rồi nhân tỉ lệ để đổi về kích thước ảnh gốc.

## pred_squares

Đầu vào: các đoạn thẳng lấy giống pred_lines.

1. Gộp đoạn: một cạnh thật hay bị detect thành nhiều đoạn nhỏ, nên gộp các đoạn nằm trên cùng một đường thẳng thành một đoạn dài bao trọn chúng.
2. Giao điểm: tìm giao điểm của mọi cặp đoạn (kéo dài đoạn ra nếu cần).
3. Lọc góc: một giao điểm được coi là góc khi 2 đoạn gần vuông góc (60-120 độ) và giao điểm nằm gần đầu mút của cả 2 đoạn. Sau đó phân loại góc: trên-trái, trên-phải, dưới-phải, dưới-trái.
4. Ghép tứ giác: 4 góc tạo thành tứ giác khi các góc liền kề có chung một đoạn thẳng (0-1, 1-2, 2-3, 3-0), tức 4 cạnh khép kín.
5. Chấm điểm: tứ giác được ưu tiên nếu cân đối (góc đối bằng nhau), to, và nằm gần giữa ảnh. Sắp xếp theo điểm, tứ giác đầu tiên là box tốt nhất.
6. Đổi toạ độ về ảnh gốc.

## Port lên Android

Chia làm 2 phần:
- Trong file .tflite: model + phần decoder (sigmoid, maxpool, topk), vì đây là phép tính trên tensor lớn, để runtime chạy cho nhanh. Em đã convert được bản này (`mlsd_tiny_decoder.tflite`), output giống bản chính thức.
- Viết lại bằng Java: pred_lines và pred_squares, vì chỉ là vòng lặp và tính hình học trên tối đa 200 điểm, rất nhẹ.

Những điểm phải chú ý khi viết lại bằng Java:
1. Phải xử lý khi không có đoạn nào / không có góc nào / không có tứ giác nào. Bản Python bị crash hoặc dùng try/except bỏ qua lỗi.
2. Bước gộp đoạn có dùng TensorFlow (maxpool, top_k) và np.unique, phải viết lại bằng vòng lặp và HashMap.
3. Chỉ số âm: numpy cho phép chỉ số âm (tự quay vòng về cuối mảng), Java thì báo lỗi, nên phải tự kiểm tra biên.
4. Hai đoạn song song thì không có giao điểm (chia cho số gần 0), phải bỏ qua.
5. Bước ghép tứ giác có 4 vòng lặp lồng nhau, có thể chậm nếu nhiều góc.
6. Dễ nhầm thứ tự toạ độ: center lưu (y, x), còn độ dời và đoạn thẳng lưu (x, y).

Cách kiểm tra bản Java đúng: lưu output của model và kết quả của Python trên vài ảnh, cho bản Java chạy trên cùng output đó rồi so kết quả.
