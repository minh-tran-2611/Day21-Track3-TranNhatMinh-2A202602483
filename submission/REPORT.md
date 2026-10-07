# Lab 21 — Evaluation Report

**Họ tên**: Trần Nhật Minh  **MSSV**: 2A202602483  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4 (Colab Free, 14.6 GB khả dụng, sm_75 → **fp16** + GradScaler, không có bf16)

> Mọi con số dưới đây lấy từ file trong `results/`. Toàn bộ NB1→NB6 chạy một lượt trên Colab
> bằng `colab/Lab21_RUN_ALL.ipynb` với `EVAL_LIMIT` để trống (đầy đủ 50 target + 15 regression),
> `EPOCHS=2`, `MASK_MODE=assistant-only`. Kết quả gatekeeper ở mục 8.

---

## 1. Lựa chọn & lý do

| | Lựa chọn | Lý do |
|---|---|---|
| Base model | `unsloth/Qwen3.5-4B` (mặc định tier T4) | Model lớn nhất vừa T4 ở bf16/fp16 LoRA (peak 8.78 GB / 14.6 GB). Giữ nguyên model mặc định để số đo so được với `docs/MEASURED-T4-2026-08-20.md`, và để baseline (b) lẫn fine-tune dùng **cùng một** base. |
| Dataset | Corpus mặc định: 250 ticket CSKH tiếng Việt → JSON 4 trường | Thang chấm khách quan (độ chính xác từng trường), không cần LLM judge. Không đổi corpus nên checksum tập eval vẫn khớp. |
| Phần cứng | Colab Free T4 | Máy cá nhân chỉ có RTX 3050 4 GB, không đủ cho 4B. |

## 2. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (`data/train_seed.jsonl`) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | **1024** (giá trị tier T4) — p95 đo được là **98** token, p99 100, max 101 → NB1 gợi ý 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch → **30** optimizer step (batch 1 × grad_accum 16 = batch hiệu dụng 16 < 32) |
| LR / scheduler | 1e-4 (10× thang full-FT), cosine, warmup 3 step |
| Độ chính xác | fp16 (T4 là Turing, không có bf16) |

**Về `max_length` lệch gợi ý.** p95 = 98 nên cả 256 lẫn 1024 đều **không cắt mẫu nào** (max = 101).
Với `per_device_batch=1` và packing tắt, không có padding giữa các chuỗi, nên đặt 1024 không tốn
thêm VRAM hay thời gian. Tôi giữ 1024 để không đổi cấu hình tier so với số đo tham chiếu. Nếu
corpus có đuôi dài hơn, tôi sẽ hạ xuống 256 theo p95.

**Template có giữ khối `<think>` không?** **Có.** `results/template_check.json`: `ok=true`,
`open_tag_present=true`, `body_present=true` — render một lượt hội thoại có trace thì trace còn
nguyên trong chuỗi. Khi sinh với `enable_thinking=False`, template tự chèn `<think>\n\n</think>`
rỗng vào cuối prompt. Ở model 4B này, phần `</think>\n\n` rơi vào **vùng được tính loss** (xem
preview mục 3), nên model học đóng khối think rỗng rồi trả JSON. Vì 250 câu trả lời đều là JSON
trần, không có trace thật nào để giữ hay che. Đó là lý do `valid_trace_rate = 0.0` ở NB5: đó là
kết quả đúng, không phải lỗi.

---

## 3. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` (mẫu ví dụ) | **0.4149** (39 / 94 token) |
| `supervised_fraction` (toàn tập train) | 9014 / 20951 = **43.0%** (log NB3) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss (giải mã ngược từ `labels != -100`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn bị che (không tính loss): toàn bộ system prompt, ticket của user và `<|im_start|>assistant\n<think>\n\n`.
Đối chứng `MASK_MODE=everything` cho 94/94 token (100%), tức là tính loss cả trên câu hỏi. Mask
`assistant-only` cắt đúng ở ranh giới assistant.

---

## 4. Ba baseline (NB2 — đo TRƯỚC khi train) và bản fine-tune (NB5)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3254.8 |
| (b) base + optimized prompt | **0.765** | 0.7911 | 1.000 | 1014.5 |
| (c) LoRA fine-tune (naive prompt) | **0.975** | **0.5222** | 1.000 | 1348.6 |

*(results/baselines_frozen.json, results/verdict.json)*

**(b) có thật sự mạnh hơn (a) không?** **Có**, 0.765 so với 0.000. Prompt ngây thơ "Phân loại
ticket sau." khiến base model viết văn xuôi thay vì JSON (format = 0.000), nên không trường nào
chấm được. Latency của (a) cao (3255 ms) cũng vì model viết dài tới `max_new_tokens`.

**Có sửa `OPTIMIZED_PROMPT` không?** **Không.** SHA `719e74d3b6232053` khớp bản gốc và
`verify.py` báo `baseline (b) prompt unmodified`. Mốc được đóng băng trước NB3.

**(b) sai ở đâu** *(results/qualitative_compare.json, chạy lại (b) và (c) trên cùng 50 ticket)*:
(b) sai 23 trường `intent`, 18 trường `urgency`, 6 trường `sentiment`. Lỗi điển hình: nhầm
"Cho tôi trả lại" (`doi_tra`) thành `hoan_tien`, và đẩy `urgency` lên `cao` khi ticket chỉ nói
"Sớm nhé". (c) chỉ sai **5 trường, cả 5 đều là `urgency`**. Không có ticket nào (c) điểm thấp hơn
(b): 33 ticket (c) thắng, 17 ticket hoà.

---

## 5. Giải phẫu cấu hình sai (NB4, chấm ở NB5 §4)

Cả bốn run có **cùng 30 step** và cùng mask, chỉ đổi **một** biến mỗi run.

| Run | Biến bị đổi | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | format | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|---|
| `correct` | — | text-linear (12 module) | 16 | 32,464,896 | 1e-4 | 0.6260 | **0.975** | 1.0 | 394.3 | 8.78 |
| `attn_only` | vị trí | q,v (2 module) | 283 | 32,456,704 | 1e-4 | 0.5376 | **0.970** | 1.0 | 261.7 | 8.79 |
| `wrong_lr` | learning rate | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.000** | 0.0 | 389.0 | 8.78 |
| `qlora` | lượng tử hoá base 4-bit | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.940** | 1.0 | 459.3 | 3.86 |

*(results/runs.csv, results/autopsy.json)*. Ngân sách `attn_only` lệch `correct` 0.025%, dưới
ngưỡng 5%; `verify.py` xác nhận đây là đối chứng công bằng.

**Xếp hạng theo target:** `correct` 0.975 > `attn_only` 0.970 > `qlora` 0.940 ≫ `wrong_lr` 0.000.
**Xếp hạng theo train loss:** `attn_only` 0.538 < `correct` 0.626 < `qlora` 0.706 < `wrong_lr` 1.570.
**Hai thứ tự khác nhau ở vị trí đầu.** Nếu xếp theo loss, ta sẽ kết luận "attention-only là
tốt nhất", nhưng trên tác vụ thật thì nó không thắng.

**5.1 — `attn_only` (cùng ngân sách tham số).** Trên tập target nó **hoà** với `correct` (0.970
so với 0.975). Chênh 0.005 tương đương đúng **1 trường trên 200**, quá nhỏ để gọi là thắng hay
thua với 50 mẫu. Train loss của nó *thấp hơn* (0.538 so với 0.626), nhưng `train_loss` của TRL
là **trung bình cả run**. Log cho thấy `attn_only` giảm nhanh hơn ở đầu (step 10: 0.826 so với
1.381), còn cuối run hai bên bằng nhau (0.0247 so với 0.0247). Thứ tự theo loss phản ánh tốc độ
hội tụ ban đầu, không phản ánh năng lực cuối. Về **rank và vị trí**: khi ngân sách đã khớp, đổi
vị trí gắn adapter (12 module → chỉ q,v với r=283) **không** thay đổi kết quả trên tác vụ hẹp này.
32 triệu tham số dư sức học một ánh xạ ticket → 4 nhãn, nên ở đây cả rank lẫn vị trí đều không
phải đòn bẩy; năng lực adapter không phải nút thắt. Tôi không khẳng định được luận điểm "§11.2
attention-only thua" từ số đo này. Để kiểm chứng cần một tác vụ khó hơn hoặc ngân sách nhỏ hơn
nhiều (ví dụ r=1–4), nơi năng lực mới thật sự khan hiếm. `attn_only` có lợi thực tế: train
nhanh hơn 34% (261.7 s so với 394.3 s) và suy luận nhanh hơn (879.5 ms so với 1348.6 ms), vì chỉ
gắn adapter vào 2 loại module.

**5.2 — `wrong_lr` (chỉ khác LR: 1e-5 thay vì 1e-4).** Loss **không phẳng**: giảm đều từ 2.163
xuống 1.119 ở step cuối (−48%), `mean_token_accuracy` lên 0.79. Nhìn riêng đường loss, tôi sẽ
kết luận "đang học, chỉ chậm, cho thêm step là được". Nhưng trên tác vụ, target = **0.000** và
format = **0.000**: với prompt ngây thơ, model chưa hề chuyển sang trả JSON, vẫn viết văn xuôi
tới hết 160 token (latency 5161 ms). Ở step 15, `correct` đã xuống 0.140 trong khi `wrong_lr` còn 1.606. Cùng 30 step, LR
thang full-FT bỏ lỡ hoàn toàn hành vi cần học. Đó là đúng Lỗi #2, và loss giảm 48% che mất điều
đó.

**5.3 — `qlora`.** VRAM đỉnh **3.86 GB so với 8.78 GB** (−4.92 GB, −56%). Cái giá phải trả:
target **0.940 so với 0.975** (−0.035, tức 7 trường trên 200), train chậm hơn 16% (459.3 s so với
394.3 s), suy luận chậm hơn 30% (1755.5 ms so với 1348.6 ms) vì phải giải lượng tử mỗi lớp. Số đo
**ủng hộ ở mức vừa phải** khuyến nghị "không dùng QLoRA cho Qwen3.5". Chất lượng tụt có thật
nhưng nhỏ, đổi lại tốc độ kém hơn. Trên T4 nơi 16-bit LoRA đã vừa (8.78/14.6 GB), không có lý do
dùng QLoRA. QLoRA chỉ đáng khi bộ nhớ là ràng buộc cứng, ví dụ base 9B trên T4.

---

## 6. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **`FAILED`**
`target Δ = +0.210` · `regression Δ = −0.269` (ngưỡng −0.020) · `valid_trace_rate = 0.0`

Bản fine-tune **thắng rõ** baseline (b) trên tác vụ đích: 0.975 so với 0.765, giữ format 1.000,
dù chỉ dùng prompt một câu thay vì prompt dài có schema và ví dụ. Nhưng nó **làm hỏng năng lực
tổng quát**: regression tụt từ 0.791 xuống 0.522, gấp hơn 13 lần ngưỡng cho phép. Đọc từng câu
trả lời *(results/qualitative_compare.json)* thấy nguyên nhân rất cụ thể: **14/15 câu hỏi phổ
thông được trả lời bằng một object JSON**, trong khi base model không trả JSON ở câu nào. Có câu
model vẫn nhét đáp án vào JSON và qua được (Hà Nội, Nguyễn Du, 100°C). Nhưng có 7 câu FT điểm
thấp hơn base. Tệ nhất là 3 câu model trả về schema triage hoặc khoá kiểu triage mà **không có
đáp án**: "1 km bằng bao nhiêu mét?" →
`{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, ...}`, "Một năm có bao nhiêu
tháng?" và "TP.HCM trước đây tên gì?". 4 câu còn lại mất keyword vì bị ép vào khoá JSON
(lời chúc sinh nhật thành `"intent": "chuc_mung_sinh_nhat"`), vì câu trả lời ngắn hơn base
(tóm tắt tục ngữ chỉ còn "kiên trì"; trái cây nhiệt đới chỉ còn "Dâu tây"), hoặc vì bị cắt ở 96
token (quang hợp).

Cơ chế nhân quả: cả 225 mẫu train đều có cùng một dạng đầu ra (JSON 4 khoá). Adapter gắn vào
**mọi** linear của decoder, train 2 epoch tới loss ≈ 0.02. Thứ model học được không chỉ là
"phân loại ticket" mà là **"mọi tin nhắn user → JSON"**. Câu hỏi regression không có system
prompt, nên không có tín hiệu nào báo đây không phải ticket. Đây là quên thảm hoạ ở tầng
*định dạng đầu ra*, đúng chẩn đoán số 2 của NB5 và deck §6.3. Không nên nới ngưỡng. Cách sửa
đúng là trộn 1–5% dữ liệu chỉ dẫn phổ thông (replay) vào tập train, rồi đo lại cả hai nhóm.

Bài toán của tôi **có lợi từ fine-tune**: +0.21 target là thật, và (b) có lỗi hệ thống
(`doi_tra`/`hoan_tien`) mà prompt khó sửa hết. Nhưng adapter hiện tại **chưa được phép thay base
model** ở một endpoint đa dụng.

---

## 7. Định tính — có cả ca THUA

Nguồn: `results/qualitative_compare.json` (target: cùng 50 ticket chấm (b) và (c); regression:
base và fine-tune trên cùng 15 câu).

| # | Đầu vào (rút gọn) | Nhãn / keyword | (b) base + prompt tối ưu | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | T#5 "…nồi chiên không dầu… Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi." | `san_pham_loi`/`thap`/`trung_tinh` | `hoan_tien`, `cao` → **0.50** | đúng cả 4 → **1.00** | ✅ FT thắng: thiếu phụ kiện là lỗi sản phẩm, không phải đòi tiền |
| 2 | T#36 "…tai nghe bluetooth… Giá bao nhiêu. Hỏi cho biết thôi. Rất thất vọng." | `hoi_thong_tin`/`thap`/`tieu_cuc` | `doi_tra`, `trung_binh` → **0.50** | đúng cả 4 → **1.00** | ✅ FT thắng: hỏi giá là hỏi thông tin |
| 3 | T#3 "…bình giữ nhiệt… Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều." | urgency = `thap` | `trung_binh` → 0.75 | `trung_binh` → **0.75** | ❌ **FT sai**: "Khi nào tiện" phải là `thap` |
| 4 | R#2 "1 km bằng bao nhiêu mét?" | keyword `1000` | base: "…**1 km = 1000 m**" → 1.0 | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, …}` → **0.0** | ❌ **FT thua**: trả schema triage, mất đáp án |
| 5 | R#13 "Thành phố Hồ Chí Minh trước đây có tên là gì?" | keyword `Sài Gòn` | base: "…**Saigon** (viết là Sài Gòn…)" → 1.0 | `{"intent": "hoi_thong_tin", "confidence": 0.95, "urgency": "trung_binh", …}` → **0.0** | ❌ **FT thua**: quên thảm hoạ dạng JSON |
| 6 | R#6 "2 mũ 10 bằng bao nhiêu?" | keyword `1024` | base giải từng bước, hết 96 token chưa ra đáp số → 0.0 | `{"intent": "math", "answer": "1024", …}` → **1.0** | ✅ FT "thắng" nhờ ngắn gọn, nhưng vẫn ở dạng JSON |

**Mẫu chung ở các ca FT thua.**
(i) Trên target, **cả 5 lỗi của FT đều là `urgency`, và cả 5 ticket đều chứa "Khi nào tiện"**
(T#3, 12, 39, 41, 46). Trong tập train, cụm này xuất hiện 30 lần và **luôn** gán `thap`. Thế mà
FT vẫn đoán `trung_binh`, giống hệt (b). Prior ngữ nghĩa của base ("khi nào tiện" nghe như một
yêu cầu có hạn) mạnh hơn 30 mẫu × 2 epoch. Đây là lỗi *underfit cục bộ*, không phải lỗi format.
(ii) Trên regression, FT thua base ở 7/15 câu (R#2, 3, 9, 11, 12, 13, 14) và chỉ thắng 1 câu
(R#6). Tất cả các ca thua có chung một nguyên nhân: model bọc câu trả lời vào JSON, và nhiều khi
dùng luôn khoá triage thay cho đáp án.

---

## 8. Kết luận & điều tôi học được

**Kết luận.** Tôi **không deploy** adapter này thay cho base model ở một endpoint chung, dù nó
thắng baseline mạnh nhất +0.21 trên tác vụ đích. Lý do là nhân quả chứ không chỉ là con số:
dữ liệu train đồng nhất 100% về định dạng đầu ra, adapter gắn vào toàn bộ decoder và train tới
loss ≈ 0.02. Ba điều đó cùng nhau dời *prior định dạng* của model cho mọi đầu vào, nên câu hỏi
phổ thông cũng bị trả bằng JSON và regression tụt 0.269. Nếu chỉ nhìn target hay loss, run này
trông như thành công trọn vẹn. Chỉ cổng hồi quy bốn nhóm mới bắt được nó.

Đòn bẩy thật sự trong lab này theo thứ tự là: (1) **learning rate**, vì một con số đổi kết quả
từ 0.975 xuống 0.000; (2) **chất lượng và thành phần dữ liệu**, vì thiếu replay gây ra FAIL, và
cụm "Khi nào tiện" với 30 mẫu nhất quán vẫn chưa sửa được prior; (3) **mask**, đã chứng minh đúng
nên không còn là biến; cuối cùng mới đến (4) **vị trí và rank adapter**. Ở ngân sách khớp, vị trí
gần như không đổi kết quả (0.975 so với 0.970), và QLoRA chỉ tốn 0.035.

Có hai đường deploy hợp lệ. Đường thứ nhất: train lại với 1–5% dữ liệu chỉ dẫn phổ thông rồi qua
lại cổng. Đường thứ hai: giữ adapter tách rời và chỉ bật nó cho request triage. NB6 đã chứng minh
hot-swap 3 adapter trên một base đang nạp, nên các request khác dùng base nguyên bản và regression
không bị ảnh hưởng. Với nhóm vận hành, đường thứ hai dùng được ngay. Đường thứ nhất mới là sửa
đúng gốc.

**Ba điều tôi học được:**
1. **`train_loss` của TRL là trung bình cả run, không phải loss cuối.** `attn_only` có loss
   "thấp hơn" `correct` chỉ vì giảm nhanh ở 10 step đầu; loss cuối của hai run bằng nhau
   (0.0247). Trước lab tôi sẽ đọc cột đó như điểm cuối và xếp hạng sai.
2. **Loss giảm 48% vẫn có thể là 0 điểm.** `wrong_lr` trông như đang học tốt, nhưng chưa chạm
   tới hành vi quyết định là "trả JSON". Từ giờ tôi luôn chấm ít nhất một vài mẫu sinh thật trước
   khi tin một đường loss.
3. **Quên thảm hoạ không ồn ào.** Model không nói sai kiến thức: nó vẫn biết Hà Nội và Nguyễn Du,
   chỉ là đổi *giọng* sang JSON cho mọi thứ. Một benchmark kiến thức chấm theo exact match sẽ chỉ
   thấy một phần; phải đọc từng output mới thấy cơ chế.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn khoảng 5% (12 mẫu) câu hỏi phổ thông có câu trả lời
văn xuôi vào tập train và chạy lại NB3 + NB5, để xem regression về lại ≥ 0.77 mà target vẫn
> 0.765 hay không. Thêm vào đó, tôi sẽ nhân bản có biến thể các mẫu "Khi nào tiện" để kiểm tra
lỗi urgency là thiếu dữ liệu hay là prior quá mạnh.

---

## 9. Ghi chú vận hành & tính toàn vẹn

- `make verify` / `scripts/verify.py`: mọi kiểm tra artefact và liêm chính đều **PASS** (mask
  proof, đủ 50 mẫu eval, prompt (b) không đổi, (b) > (a), checksum tập eval, 4 run chung 30 step,
  `attn_only` khớp ngân sách 32,456,704 so với 32,464,896). Cảnh báo duy nhất là phán quyết
  FAILED, đã phân tích ở mục 6.
- **NB6** đo điểm trước và sau merge: 0.975 → 0.975 (Δ +0.000, ngưỡng 0.01), nên assert đạt.
  Bước ghi checkpoint merge 9 GB ra đĩa bị treo khoảng 16 phút vì RAM host Colab đầy
  (10.7/12.7 GB), nên tôi dừng NB6 ở đó. `results/merge_check.json` ghi đúng hai số đã đo kèm ghi
  chú `note`, và **không** lưu checkpoint merge. Phần hot-swap chạy lại trong một ô riêng:
  `results/hotswap.json` cho thấy 3 adapter (`correct`, `attn_only`, `qlora`) nạp trên cùng một
  base, cả ba cho cùng nhãn đúng với ticket #0. Mã của ô bổ sung này nằm trong
  `submission/colab_extra_cell.py`.
- `results/qualitative_compare.json` cũng do ô bổ sung đó tạo ra: chạy lại (b) và (c) trên cùng
  tập eval. Vì greedy decode nên các số khớp chính xác NB2/NB5: (b) 0.765, (c) 0.975, regression
  0.7911 → 0.5222.
- Adapter không nộp kèm (bài nộp theo Option C — code + `results/`). Phiên bản thư viện là bản
  Colab cài từ `requirements.txt` của repo tại commit `d27c1c0`.

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + assert điểm không tụt (Δ 0.000) + hot-swap 3 adapter — checkpoint merge không lưu được do RAM host, xem mục 9
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
