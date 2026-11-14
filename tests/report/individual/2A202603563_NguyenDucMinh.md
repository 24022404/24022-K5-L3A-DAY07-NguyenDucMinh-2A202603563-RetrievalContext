# Báo cáo Day 7

Nguyễn Đức Minh · 2A202603563 · L3A

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** dùng lại corpus Day 6: *The Elements of Style* (W. Strunk Jr., 1920), Project Gutenberg #37134, thuộc phạm vi công cộng ở Mỹ. Gồm 47 đơn vị trang (mỗi đơn vị ≥1.500 ký tự) và **188 chunk** của cấu hình C Day 6 (chia theo đoạn, tối đa 800 ký tự). Chunk được xuất trực tiếp từ collection Chroma (`src/export_day6.py` → `data/chunks.json`).
- **Vector store, embedding model, phiên bản và số chiều:** Chroma 1.5.9 `PersistentClient` (thư mục `chroma_db/` của repo Day 6, collection `eos_C`), HNSW, metric cosine. Embedding là `sentence-transformers/all-MiniLM-L6-v2` (384 chiều, chuẩn hóa L2), giống hệt Day 6. Index được dựng lại bằng `python src/run.py` của Day 6 và cho cùng kết quả (dense dev 9/16, 13/16).
- **Agent model, cách chạy và phân chia dev/test:** Gemini `gemini-3.5-flash-lite` qua Gemini API, gói miễn phí, dùng function calling (`google-genai`). Benchmark có **25 câu**:
  - dev: 16 câu có đáp án và 2 câu thiếu bằng chứng (u01, u02),
  - test: 5 câu có đáp án và 2 câu thiếu bằng chứng (u03, u04).
  
  21 câu có đáp án lấy lại từ Day 6. Split giữ nguyên như Day 6; tập test Day 6 chưa từng dùng để chỉnh Day 7.
- **Cách xác định câu có/thiếu bằng chứng và tiêu chí chấm:**
  - Câu có đáp án: `gold` là đoạn evidence nằm trọn trong một chunk đúng trang. `export_day6.py` báo lỗi nếu không tìm được.
  - Câu thiếu bằng chứng: hỏi sát chủ đề (dấu gạch ngang, *irregardless*, số dưới mười, trích dẫn website). Đã kiểm tra bằng cách tìm từ khóa trong toàn corpus (0 kết quả) và đọc các chương liên quan (`absence_reason` trong `data/cases.json`).
  - Tiêu chí chọn cấu hình, chốt trước khi chạy test: **Context Evidence Rate trên dev**, sau đó Evidence Hit@3, rồi Hit@1, rồi độ trễ.

## Retrieval và thử nghiệm

Bảng dưới chạy với query gốc, top-3, cùng corpus, index và nhãn. Cách đóng gói context giống `starter/baseline.py`: giữ thứ tự hạng và bỏ chunk không vừa ngân sách ký tự (tính cả dòng header ID/nguồn/trang). Chi phí 0 USD vì mọi thứ chạy local.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline `dense` | Dense Chroma Day 6, top-3, filter chương nếu câu hỏi có, 800 ký tự | Baseline | dev | 9/16 (56%) | 13/16 (81%) | 8/16 (50%) | 30 ms/câu · 0 USD |
| Phương pháp 2 `bm25` | BM25 (`rank-bm25`, tách từ `\w+`), top-3, 800 | Từ khóa bắt được cụm nguyên văn | dev | 9/16 (56%) | 11/16 (69%) | 7/16 (44%) | 2 ms · 0 |
| Phương pháp 3 `hybrid` | RRF (k=60) của dense top-20 và BM25 top-20, top-3, 800 | Gộp hai cách bù lỗi cho nhau | dev | 9/16 (56%) | 12/16 (75%) | 8/16 (50%) | 29 ms · 0 |
| Cải thiện 1 `hybrid_rerank` | Hybrid top-20, xếp lại bằng cross-encoder `ms-marco-MiniLM-L-6-v2`, top-3, 800 | Rerank sửa các câu diễn đạt khác tài liệu (q04) | dev | 11/16 (69%) | 14/16 (88%) | 8/16 (50%) | 894 ms · 0 |
| Cải thiện 2 `hybrid_rerank_1600` | Như trên, ngân sách **1600** | Bằng chứng tìm được nhưng bị loại khi đóng gói 800 ký tự | dev | 11/16 (69%) | 14/16 (88%) | **14/16 (88%)** | 952 ms · 0 |
| Cấu hình đã chọn | `hybrid_rerank_1600` | | test | 4/5 (80%) | 5/5 (100%) | 5/5 (100%) | 782 ms · 0 |
| Baseline | `dense` | | test | 3/5 (60%) | 5/5 (100%) | 4/5 (80%) | 26 ms · 0 |

`hybrid_rerank` vừa là phương pháp thứ 4 (có thêm bước rerank) vừa là cải thiện 1. Cải thiện 2 chỉ đổi ngân sách context.

**Hai lỗi trên dev:**
1. **q04 (câu diễn đạt khác tài liệu):** người dùng hỏi về "interrupting phrase… one side", còn sách viết "parenthetic… never insert one comma and omit the other". Dense và BM25 đều trượt top-3. *Giả thuyết:* cross-encoder đọc đồng thời câu hỏi và đoạn văn nên hiểu được cách diễn đạt khác. *Kết quả:* với rerank, q04 vào top-3 và Hit@3 tăng từ 13 lên 14/16. Tuy vậy, ở ngân sách 800, q04 vẫn bị loại khi đóng gói.
2. **Mất bằng chứng khi đóng gói (q07, q09, q10, q11, q13 với dense; q04, q07, q09–q11, q13 với rerank 800):** chunk đúng nằm trong top-3 nhưng dài 729–771 ký tự, cộng header là vượt 800 nên bị bỏ. *Giả thuyết:* nâng ngân sách lên 1600 thì giữ được. *Kết quả:* Context Evidence Rate tăng từ 8 lên 14/16. Đổi lại, context trung bình dài hơn (594 → 1212 ký tự) và tốn thêm token.
3. Còn trượt q08 (phải nhận ra một ví dụ sai, không phải tra quy tắc) và q16 (lọc chương V nhưng rerank xếp mục *While* ngoài top-3).

Cấu hình được chọn theo quy tắc đã chốt: `hybrid_rerank_1600` có Context Evidence Rate trên dev cao nhất. Sau khi chốt mới chạy test.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Có một tool `search_documents(query)`. Gemini tự viết query và tự quyết định có tìm tiếp hay không, tối đa **3 lượt tìm**.
  - Context được cộng dồn qua các lượt, bỏ chunk trùng, đóng gói theo ngân sách của cấu hình. Mỗi lượt tìm trả về context hiện tại cho model.
  - Model phải trích ID chunk trong `[...]` hoặc trả lời `INSUFFICIENT_EVIDENCE`. Nếu citation không có trong context hoặc không có citation thì hệ thống chuyển thành từ chối.
  - Hết 3 lượt tìm thì code ép model trả lời; nếu model vẫn không trả lời thì ghi là từ chối.
  - Lỗi 429/503 được thử lại tối đa 5 lần (chờ tăng dần); lỗi khác ghi `status: error`, không thay bằng kết quả giả. Gọi tối đa 12 request/phút để nằm trong giới hạn miễn phí.
  - Filter chương chỉ lấy từ câu hỏi (q14–q16, q20), không lấy từ gold.

| Cấu hình / split | Tool use | Số lượt gọi (1/2/3) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| dense / dev | 18/18 | 6/2/10 | 10/0/6 → **0,75** | 7/7 | 7/7 | 2/2 | 9/16 |
| hybrid_rerank_1600 / dev | 17/17 (1 lỗi mạng) | 12/0/5 | 4/0/11 → **1,47** | 11/11 | 11/11 | 2/2 | 4/15 |
| dense / test | 7/7 | 5/2/0 | 0/0/5 → **2,00** | 6/6 | 5/5 | 2/2 | 0/5 |
| hybrid_rerank_1600 / test | 7/7 | 5/2/0 | 1/0/4 → **1,60** | 6/6 | 5/5 | 2/2 | 0/5 |

Các số trên được tổng hợp bằng `starter/agent_metrics.py` từ nhãn chấm tay (`results/agent/summary_*.json`).

**Chấm context trên test (5 câu có đáp án, thang 0/1/2):**
- dense: 2/2/2/2/2. Riêng q17 dùng bằng chứng tương đương ở trang 28, không phải câu gold ở trang 27.
- hybrid_rerank_1600: 2/2/**1**/2/2. q19 chỉ có nửa đầu mục *However*, thiếu câu trả lời ở chunk p37-000.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q12 (dev, hybrid_rerank_1600) | Có bằng chứng | 1 lần tìm `revised out of every sentence in which it occurs`, top-1 là p19-002 | "The expression 'the fact that' should be revised out of every sentence…" [p19-002] | 2: đúng; citation hỗ trợ. Dense thì trượt câu này qua cả 3 lượt tìm |
| u04 (test, cả 2 cấu hình) | Thiếu bằng chứng | 2 lần tìm (`website online article…`, `website internet online`), chỉ ra mục References về act/scene/page | `INSUFFICIENT_EVIDENCE: The book does not contain instructions on how to cite a website…` | Từ chối đúng, không bịa quy tắc |
| q19 (test, hybrid_rerank_1600) | Lỗi | 1 lần tìm `however first in its sentence`, top-3 có p36-007 (nửa đầu mục *However*), thiếu p37-000 | "When however means 'nevertheless,' it should not come first…" [p36-007] | 0: câu trả lời không trả lời câu hỏi. Mục từ bị cắt qua 2 trang (lỗi đã thấy từ Day 6) và query agent viết lại làm mất chữ "mean" |

- **Trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Trace: `results/retrieval/*.json` và `results/agent/{config}_{split}.json` (mỗi tool call có query, filter, kết quả, score, context_ids, câu trả lời, citation, token, thời gian).
  - Nhãn: `results/agent/reviews_*.json`. Thống kê: `stats_*.json` và `summary_*.json`.
  - Lệnh chạy: xem README.
  - Thời gian trung vị mỗi câu: dense 19,7 s và hybrid_rerank_1600 10,1 s trên dev; khoảng 10 s trên test (phần lớn do chờ giới hạn tốc độ).
  - Token trên dev: dense 30.484 vào / 1.670 ra; hybrid_rerank_1600 26.914 / 1.585.
  - Chi phí **0 USD** (gói miễn phí).
  - Nhãn do AI (Claude) đề xuất, tác giả đọc lại và xác nhận.
- **Kết quả trước/sau, đánh đổi và giới hạn:**
  - Trên dev, cấu hình chọn tốt hơn hẳn baseline: grounded 0,75 → 1,47, từ chối nhầm 9/16 → 4/15, và cần ít lượt tìm hơn (12/17 câu chỉ tìm 1 lần). Với dense, agent thường tìm tới 3 lần rồi từ chối vì bằng chứng bị loại khi đóng gói 800 ký tự.
  - Trên test (chỉ 5 câu), baseline lại hơn: 2,00 so với 1,60, do một lỗi ở q19. Kết quả retrieval trên test với query gốc vẫn ủng hộ cấu hình chọn (Hit@1 4/5 so với 3/5, context 5/5 so với 4/5), nhưng agent tự viết lại query nên kết quả khác. Với 5 câu, chênh 1 câu là trong phạm vi nhiễu; không thể kết luận cấu hình nào tốt hơn một cách tổng quát.
  - Đánh đổi: rerank chậm hơn khoảng 30 lần (≈0,9 s so với 0,03 s mỗi lượt tìm, chạy CPU), context gấp đôi.
  - Lỗi còn lại: context cộng dồn ưu tiên chunk của lượt trước, nên bằng chứng tìm ra ở lượt 2–3 có thể không vừa (q07, q16 với hybrid). Starter README cũng đã cảnh báo điều này.
- **Sự cố khi chạy (ghi nhận trung thực):**
  - Lần chạy đầu dùng `gemini-2.5-flash` và hết hạn mức 20 request/ngày của gói miễn phí (lỗi 429). Mình đổi sang `gemini-3.5-flash-lite`; **mọi kết quả trong báo cáo đều dùng model này**.
  - `config.py` ban đầu đọc tên model trước khi nạp `.env`; đã sửa.
  - Trong lần chạy dev đầu, agent lỗi "loop ended without an answer" (6/18 câu với dense, 3/18 với hybrid) vì model vẫn gọi tool sau khi hết lượt. Đã sửa `agent.py` (ép trả lời, nếu không thì ghi là từ chối) và chạy lại dev.
  - Test được chạy trước khi sửa lỗi này và không chạy lại, vì không câu test nào vượt giới hạn lượt tìm nên đoạn code đã sửa không ảnh hưởng tới chúng.
  - Một lỗi mạng (`WinError 10053`) ở hybrid dev q08 được báo riêng và không tính vào chất lượng.
- **Giới hạn khác:**
  - Gemini có thể cho kết quả hơi khác giữa các lần chạy dù đặt temperature 0.
  - Regex citation không nhận dạng được trường hợp gộp nhiều ID trong một ngoặc `[a, b]`, nên một số citation phụ không được tính.
  - Chỉ có một người chấm (có AI hỗ trợ).
  - Tập test nhỏ.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.
