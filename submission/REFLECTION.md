# Reflection — Lab 21

*Trần Nhật Minh · 2A202602483*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Run `wrong_lr` làm loss giảm từ 2.163 xuống 1.119 (−48%) mà target vẫn đúng **0.000**, format
cũng 0.000. Tôi vẫn nghĩ "LR sai thì loss phẳng". Thực tế loss không phẳng, chỉ là chưa đi tới
hành vi quyết định (trả JSON). Điều thứ hai làm tôi bất ngờ: `attn_only` với r=283 hoà
`correct` trên target (0.970 so với 0.975), trong khi tôi đã chờ nó thua như slide nói.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải chỗ tôi dự đoán. Train chỉ mất khoảng 6–8 phút mỗi run. Thời gian lại tốn vào
**vận hành**: tải 9.32 GB trọng số, sinh văn bản cho ba bộ baseline, và nhất là NB6 treo khoảng
16 phút khi ghi checkpoint merge 9 GB vì RAM host Colab đầy (10.7/12.7 GB). Rồi tôi phải tìm
cách lấy file `results/` từ Colab về máy mà không tải trọng số. Phần "ML" của lab ngắn hơn phần
"hạ tầng" rất nhiều.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi tin "fine-tune thắng baseline trên tác vụ = fine-tune tốt". Bản của tôi thắng prompt tối ưu
+0.21 nhưng trả lời 14/15 câu hỏi phổ thông bằng JSON, nên regression tụt 0.269. Thắng target là
điều kiện cần, chưa phải điều kiện đủ. Tôi cũng từng đọc cột `train_loss` như loss cuối; thật ra
nó là trung bình cả run, và chính vì thế `attn_only` trông "tốt hơn" `correct`.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Claude Code (agent) để điều khiển Colab qua Chrome: chạy RUN_ALL, thêm ô bổ sung cho
NB6, lấy các file `results/` về repo, kiểm tra lại toàn bộ điểm từ dự đoán, rồi soạn REPORT. Nó
sai hoặc vấp ở mấy chỗ:
(a) Ban đầu nó định cài torch CUDA và chạy trên laptop RTX 3050 4 GB của tôi, nên tôi phải dừng
lại và bảo chạy trên Colab.
(b) Nó chạy NB6 nguyên bản mà không lường trước RAM host Colab không đủ để ghi checkpoint merge
9 GB, dẫn tới treo và phải ngắt.
(c) Bản nháp report có hai chỗ diễn đạt sai số liệu (gọi nhầm step 10 thay vì step 15, và nói cả
7 câu regression thua đều là schema triage). Các lỗi này được bắt khi đối chiếu lại với log và
`qualitative_compare.json`.
Bài học: số nào AI viết ra cũng phải đối chiếu với file `results/`.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Dựng **tập eval đóng băng hai nhóm** trước khi chạm vào dữ liệu train: tác vụ đích của khách hàng
và một tập hồi quy về những gì model *đang* làm tốt mà khách hàng vẫn cần. Sau đó đo baseline
prompt tốt nhất có thể. Lab này cho thấy nếu thiếu nhóm hồi quy, tôi đã ship một model trả JSON
cho mọi câu hỏi mà vẫn tưởng mình thắng.
