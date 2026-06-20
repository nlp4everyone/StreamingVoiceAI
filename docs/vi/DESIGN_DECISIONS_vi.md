# Quyết định thiết kế

Tài liệu này giải thích lý do đằng sau mỗi lựa chọn kiến trúc quan trọng, kèm phân tích ưu/nhược điểm và các phương án thay thế đã cân nhắc.

---

## 1. Audio Buffer — Ring Buffer

### Mô tả

Âm thanh đến từ client được lưu vào một **ring buffer** có dung lượng cố định (mặc định 12 giây). Khi buffer đầy, dữ liệu cũ nhất bị ghi đè thay vì cấp phát thêm bộ nhớ. Mọi inference window đều được đọc từ buffer này dưới dạng snapshot.

### Ưu điểm

- **Bộ nhớ cố định và dự đoán được:** Không cấp phát động trong runtime — không gây GC pause dưới tải cao.
- **Truy xuất linh hoạt:** Có thể lấy bất kỳ khoảng thời gian nào trong buffer (N giây gần nhất, hoặc một khoảng tùy ý) mà không cần copy dư thừa.

### Nhược điểm

- **Dung lượng cố định:** Nếu inference window lớn hơn dung lượng buffer, window sẽ bị cắt ngắn mà không có cảnh báo rõ ràng.
- **Giả định single-writer:** Không có cơ chế đồng bộ hóa — an toàn với một luồng nhận, nhưng cần xem xét lại nếu mở rộng thêm nguồn ghi.

### Thay thế đã cân nhắc

| Phương án | Lý do không chọn |
|---|---|
| Queue lưu từng gói âm thanh | Không hỗ trợ truy xuất theo time range; tốn bộ nhớ hơn |
| Ghi file tạm rồi đọc lại | Latency cao; phức tạp không cần thiết |

---

## 2. VAD Model — Silero VAD

### Mô tả

Hệ thống dùng **Silero VAD** — mô hình học sâu nhỏ gọn được export sang ONNX, chạy qua ONNX Runtime mà không cần PyTorch. Mô hình phân tích từng frame ~32ms và trả về **xác suất có giọng nói theo từng frame**, thay vì chỉ trả kết quả nhị phân.

### Ưu điểm

- **Xác suất theo từng frame:** Đây là yêu cầu bắt buộc của Speech Trimmer — cần bản đồ xác suất để xác định chính xác điểm bắt đầu/kết thúc giọng nói trong window, không chỉ biết "có hay không".
- **Không cần PyTorch ở runtime:** Chạy qua ONNX Runtime — giảm đáng kể dung lượng Docker image và thời gian khởi động so với load full PyTorch.
- **Độ chính xác cao:** Vượt trội so với các phương pháp rule-based trong nhiều môi trường âm thanh, đặc biệt với giọng có accent và nhiễu nền nhẹ.
- **Hỗ trợ INT8 quantization:** Có thể quantize khi khởi động lần đầu, giảm ~50% RAM và tăng tốc inference trên CPU.
- **Model nhỏ:** ~1–2 MB — bake vào Docker image mà không ảnh hưởng đáng kể đến kích thước image.

### Nhược điểm

- **Latency cao hơn rule-based:** Inference qua neural network chậm hơn WebRTC VAD, dù đã được giảm thiểu bằng VAD pool và ONNX Runtime.
- **Black box:** Không thể giải thích tại sao một frame cụ thể bị đánh giá là silence — khó debug khi VAD cho kết quả sai trên giọng nói bất thường.

### Thay thế đã cân nhắc

| Phương án | Lý do không chọn |
|---|---|
| WebRTC VAD (py-webrtcvad) | Rule-based, chỉ trả kết quả nhị phân — không có xác suất theo frame, Speech Trimmer không dùng được |
| PyAnnote Audio VAD | Chính xác cao nhưng phụ thuộc PyTorch nặng; không cần thiết khi Silero đã đủ |
| Whisper VAD | Tích hợp sẵn trong Whisper nhưng không tách riêng được; quá nặng cho một bước pre-filter |
| Energy-based VAD (RMS threshold) | Đơn giản nhất nhưng kém chính xác khi có nhiễu nền; không trả xác suất theo frame |

---

## 3. VAD Pool

### Mô tả

Thay vì dùng một model VAD duy nhất cho tất cả session, hệ thống khởi tạo sẵn một **pool** gồm nhiều instance VAD (mặc định 8 instance). Khi cần phát hiện giọng nói, một instance được lấy ra từ pool, chạy trên thread riêng, rồi trả lại ngay sau khi xong.

### Ưu điểm

- **Xử lý song song thực sự:** Nhiều session có thể chạy VAD cùng lúc mà không tranh chấp nhau.
- **Tái sử dụng xác suất frame:** Xác suất từng frame đã tính cho quyết định VAD được dùng lại trực tiếp bởi Speech Trimmer — không cần chạy ONNX lần thứ hai.
- **Backpressure khi pool cạn kiệt:** Khi toàn bộ instance đang bận, hệ thống gửi tín hiệu `backpressure` thay vì queue vô hạn hoặc block luồng nhận.

### Nhược điểm

- **Tăng startup time:** Tải nhiều instance VAD khi khởi động server làm chậm thời gian sẵn sàng.
- **Pool exhaustion:** Khi toàn bộ instance đang bận, inference window mới bị drop và backpressure được gửi về client.
- **Bộ nhớ tuyến tính theo pool size:** Mỗi instance tốn ~10–20 MB RAM; pool 8 instance chiếm 80–160 MB chỉ riêng cho VAD.

### Thay thế đã cân nhắc

| Phương án | Lý do không chọn |
|---|---|
| Một instance dùng chung + lock | Bottleneck nghiêm trọng khi nhiều session đồng thời |
| PyTorch SileroVAD | Dependency nặng; không cần thiết |
| WebRTC VAD | Kém chính xác hơn; không trả xác suất theo frame |

---

## 4. VAD Trigger Strategies

### Mô tả

VAD trả về xác suất có giọng nói theo từng frame (~32ms). Một **trigger strategy** chuyển chuỗi xác suất đó thành quyết định nhị phân có/không có tiếng nói cho toàn bộ window. Có ba chiến lược:

- **Consecutive frames:** Chỉ kết luận là giọng nói khi N frame liên tiếp đều vượt ngưỡng.
- **EMA smoothed:** Làm mịn xác suất qua trung bình trọng số trước khi so ngưỡng — giảm false positive từ tiếng ồn ngắn.
- **State machine** *(mặc định qua settings.yaml)*: FSM với ngưỡng bắt đầu (onset) cao hơn ngưỡng kết thúc (offset) — hysteresis ngăn nhảy trạng thái liên tục khi xác suất dao động quanh ngưỡng.

### Ưu điểm

- **Cấu hình được mà không cần sửa code** — phù hợp khi triển khai ở các môi trường âm thanh khác nhau.
- **EMA cân bằng tốt** giữa độ nhạy và khả năng chống nhiễu.
- **State machine phù hợp môi trường nhiễu cao** — hysteresis ngăn chuyển trạng thái liên tục.

### Nhược điểm

- **Consecutive frames** nhạy với tiếng ồn xung (click, pop) nếu N quá thấp.
- **EMA** tạo độ trễ nhỏ ở đầu và cuối câu nói do quán tính của trung bình trượt.
- **State machine** cần tune hai ngưỡng riêng biệt — khó cấu hình đúng từ ban đầu.

---

## 5. Adaptive Inference Pacing

### Mô tả

Thay vì chạy inference theo chu kỳ cố định, hệ thống tự động điều chỉnh tốc độ inference cho từng session:

- **`ONSET_INTERVAL_MS` (400ms)** — dùng ngay khi phát hiện tiếng nói mới; ưu tiên cập nhật partial transcript nhanh.
- **`STABLE_INTERVAL_MS` (1200ms)** — dùng khi transcript không thay đổi qua nhiều window liên tiếp; giảm ASR call dư thừa.

Interval được reset về `ONSET_INTERVAL_MS` mỗi khi `vad_state.last_speech_time` tiến lên (phát hiện frame nói mới).

### Ưu điểm

- **Giảm ASR call dư thừa** khi transcript đã ổn định — ít GPU call hơn, giảm tải NeMo.
- **Partial transcript nhanh khi cần** — onset window giữ latency thấp ở đầu mỗi câu nói.
- **Granularity per-session** — mỗi session theo dõi `current_interval_ms` riêng, không ảnh hưởng lẫn nhau.

### Nhược điểm

- **Hai tham số cần tune** thay vì một (`ONSET_INTERVAL_MS` và `STABLE_INTERVAL_MS`), có thể gây khó hiểu ban đầu.
- **Phát hiện stability đơn giản** — chỉ so sánh output hiện tại với `last_partial_for_stability`; không xét đến các thay đổi nhỏ ở đuôi câu.

### Thay thế đã cân nhắc

| Phương án | Lý do không chọn |
|---|---|
| Chỉ dùng fixed interval | Lãng phí ASR call khi transcript ổn định; hoặc nếu interval dài thì partial transcript bị chậm |
| Backoff dựa theo độ sâu queue | Phức tạp hơn; queue depth là tín hiệu trễ so với transcript stability |

---

## 6. Trailing-Silence Window Correction

### Mô tả

Cuối mỗi câu nói xảy ra một vấn đề: người dùng đã dừng nói, nhưng inference window 6 giây vẫn còn chứa các frame nói cũ từ trước — khiến `is_speech` trả về `True`. Điều này kích hoạt các ASR call không cần thiết trên window toàn im lặng.

Cơ chế correction ghi đè `is_speech=True` thành `False` khi speech segment cuối cùng trong window kết thúc cách đây hơn `TRAILING_SILENCE_MS` (1000ms) — thực hiện trước khi `VADState.update()` được gọi.

### Ưu điểm

- **Giảm ~50% ASR call ở cuối câu** — vị trí thường xảy ra stale VAD nhất.
- **Không tốn thêm ONNX inference** — chỉ là một phép kiểm tra Python thuần trên `vad_state.last_speech_time`, tái dùng dữ liệu đã có.

### Nhược điểm

- **Có thể bỏ qua ASR khi có khoảng dừng ngắn giữa câu** nếu `TRAILING_SILENCE_MS` đặt quá thấp — các từ cách nhau bởi khoảng nghỉ tự nhiên có thể bị cắt.
- **Phụ thuộc vào `last_speech_time` chính xác** — nếu VAD đọc sai frame nói cuối, ranh giới correction cũng bị lệch theo.

---

## 7. Speech Trimming + MIN_TRIMMED_AUDIO_MS Gate

### Mô tả

Trước khi gọi ASR, inference window 6 giây được cắt về đúng vùng có tiếng nói, dựa trên xác suất VAD từng frame đã tính sẵn (không chạy ONNX lần thứ hai). Audio sau khi cắt được thêm `SPEECH_PADDING_MS` (200ms) hai bên để giữ ngữ cảnh.

Sau khi cắt, một length gate kiểm tra: nếu kết quả ngắn hơn `MIN_TRIMMED_AUDIO_MS` (500ms), window bị bỏ qua hoàn toàn mà không gọi ASR.

### Ưu điểm

- **Giảm kích thước input ASR** — audio ngắn hơn → NeMo inference nhanh hơn và ít overhead mạng hơn.
- **Tái dùng VAD probs** — `segments_from_probs()` là Python thuần; không cần chạy ONNX lần thứ hai.
- **MIN_TRIMMED_AUDIO_MS gate ngăn ASR call gần như im lặng** — tránh gửi window mà VAD chỉ phát hiện một blip nhỏ xác suất nói.

### Nhược điểm

- **Trimming có thể thất bại** nếu xác suất VAD nhiễu — fallback là dùng toàn bộ window, an toàn nhưng mất đi lợi ích của trimming.
- **MIN_TRIMMED_AUDIO_MS là ngưỡng cứng** — một từ ngắn thực sự như "ừ" gần ranh giới 500ms có thể bị bỏ qua.

---

## 8. Global ASR Semaphore

### Mô tả

Một `asyncio.Semaphore` duy nhất (`ASR_SEMAPHORE_LIMIT`, mặc định 8) giới hạn tổng số NeMo HTTP request đồng thời trên **tất cả session**. Mỗi inference worker phải acquire semaphore trước khi gọi ASR.

### Ưu điểm

- **Bảo vệ NeMo khỏi quá tải** — ngăn lượng session tăng đột biến flood ASR server cùng lúc.
- **Shared cap hiệu quả hơn per-session limit** — session đang idle hoặc chậm không chiếm quota; session bận có thể dùng nhiều hơn.

### Nhược điểm

- **Tranh chấp toàn cục khi tải cao** — các session cùng cạnh tranh một semaphore; session đang xử lý audio dài có thể làm chậm session khác.
- **Không có priority** — tất cả session ngang nhau; không có cơ chế ưu tiên session mới hơn hoặc gần finalize hơn.

### Thay thế đã cân nhắc

| Phương án | Lý do không chọn |
|---|---|
| Per-session semaphore | Kém hiệu quả — session idle vẫn chiếm quota; không bảo vệ được khi nhiều session cùng tăng đột biến |
| Không có semaphore (concurrency không giới hạn) | NeMo server sẽ quá tải khi số session cao |

---

## 9. Stabilization Pipeline

### Mô tả

ASR streaming liên tục trả về hypothesis mới — và hypothesis mới đôi khi ngắn hơn hoặc khác với lần trước, gây hiện tượng **rollback** (văn bản hiển thị bị thu ngắn). Pipeline ổn định hóa gồm hai tầng:

**Tầng 1 — Longest Common Prefix (LCP):**
So sánh hypothesis mới với hypothesis trước và chỉ giữ lại phần prefix chung. Phần đuôi còn biến động được giữ lại, chưa hiển thị.

| Chế độ | Cơ chế | Khi nào phù hợp |
|---|---|---|
| **Word-level** *(mặc định)* | So sánh theo từng từ (tách bởi khoảng trắng). Ví dụ: `"xin chào bạn"` vs `"xin chào anh"` → prefix chung: `"xin chào"` | Tiếng Việt và các ngôn ngữ dùng khoảng trắng làm ranh giới từ |
| **Character-level** | So sánh theo từng ký tự. Ví dụ: `"xin chào bạn"` vs `"xin chào anh"` → prefix chung: `"xin chào "` | Ngôn ngữ không dùng khoảng trắng (tiếng Trung, tiếng Nhật) hoặc khi cần độ mịn cao hơn |

Word-level an toàn hơn — không bao giờ cắt giữa chừng một từ, tránh hiển thị nửa từ rồi biến mất ngay.

**Tầng 2 — Rollback Suppression:**
Bảo vệ văn bản đã hiển thị khỏi bị thu ngắn. Có nhiều chiến lược:

| Chiến lược | Cơ chế | Khi nào phù hợp |
|---|---|---|
| **Frozen prefix** *(mặc định)* | Đóng băng dần prefix sau N hypothesis nhất quán; từ chối hypothesis mâu thuẫn với vùng đã đóng băng | Cân bằng tốt cho tiếng Việt |
| **Hard length** | Số từ chỉ tăng, không bao giờ giảm | Khi pipeline downstream không chấp nhận xóa từ |
| **Edit distance** | Từ chối hypothesis nếu khoảng cách chỉnh sửa vượt ngưỡng | Môi trường nhiễu cao |
| **N-consecutive** | Chỉ commit rollback sau N lần liên tiếp đồng thuận | Khi ưu tiên độ chính xác hơn latency |

### Ưu điểm

- **Mỗi session có trạng thái riêng:** Stabilizer được reset sau mỗi lần `finalize()` — không bị ảnh hưởng giữa các session hay giữa các câu.
- **Modular:** Các chiến lược độc lập và có thể nối chuỗi qua `StabilizerPipeline`.
- **Phù hợp tiếng Việt:** Word-level LCP phù hợp với cách tokenize tiếng Việt dùng khoảng trắng.

### Nhược điểm

- **Tăng latency hiển thị:** Ngưỡng đóng băng quá thận trọng khiến văn bản xuất hiện chậm hơn thực tế.
- **Không tự phục hồi khi ASR sai nghiêm trọng:** Một số chiến lược (ví dụ `hard_length`) khóa trạng thái sai và không thể sửa lại nếu không reset session.

---

## 10. Per-Session Inference Worker

### Mô tả

Mỗi WebSocket session có một **worker riêng** chạy nền (`asyncio.Task`), chuyên drain hàng đợi inference. Luồng nhận âm thanh chỉ đẩy window vào hàng đợi — không bao giờ chờ VAD hay ASR hoàn thành.

### Ưu điểm

- **Luồng nhận không bao giờ bị block:** Dù ASR mất 2–3 giây, client vẫn tiếp tục gửi âm thanh bình thường và buffer tiếp tục được cập nhật.
- **Backpressure rõ ràng:** Khi hàng đợi đầy, hệ thống gửi tín hiệu `backpressure` về client thay vì im lặng bỏ qua hoặc block.
- **Isolation giữa các session:** Lỗi hay timeout ở một session không ảnh hưởng các session khác.

### Nhược điểm

- **Queue drift:** Nếu ASR chậm hơn tốc độ tạo inference window, hàng đợi tích lũy các window cũ — transcript có thể bị trễ so với giọng nói thực tế.
- **Không có priority queue:** Window được xử lý theo FIFO; không có cơ chế ưu tiên window mới hơn khi đang tắc nghẽn.

### Thay thế đã cân nhắc

| Phương án | Lý do không chọn |
|---|---|
| Xử lý inline trong luồng nhận | Block hoàn toàn việc nhận âm thanh khi ASR chậm |
| Thread riêng mỗi session | Overhead cao hơn; không cần thiết với async I/O |
| Process riêng mỗi session | Quá nặng; latency giao tiếp giữa process cao |

---

## 11. ASR Engine — External HTTP Service (NVIDIA NeMo)

### Mô tả

ASR chạy như một **external service** tách biệt. Streaming service giao tiếp với NeMo server qua HTTP, gửi audio dưới dạng WAV và nhận về chuỗi transcript. Kết nối HTTP được tái sử dụng qua connection pooling (`aiohttp.ClientSession`).

### Ưu điểm

- **Tách biệt hoàn toàn:** ASR engine có thể được thay thế (Whisper, Google STT, v.v.) mà không ảnh hưởng phần còn lại của hệ thống.
- **Scale độc lập:** NeMo server có thể được scale riêng (số GPU, số replica) tùy theo nhu cầu ASR, độc lập với streaming service.
- **Timeout cứng:** Mỗi request có timeout rõ ràng (`ASR_CONNECT_TIMEOUT`, `ASR_REQUEST_TIMEOUT`) — ASR chậm bị hủy, tránh treo session vô thời hạn.

### Nhược điểm

- **Phụ thuộc mạng:** NeMo server down hoặc mạng chậm làm toàn bộ ASR pipeline ngừng hoạt động. Chưa có fallback hay circuit breaker.
- **Không có retry:** Request thất bại bị bỏ qua mà không thử lại.

### Thay thế đã cân nhắc

| Phương án | Lý do không chọn |
|---|---|
| gRPC streaming | Latency thấp hơn, nhưng phức tạp hơn nhiều trong khi HTTP API đã đủ dùng |
| Nhúng model trực tiếp vào service | Tăng RAM đáng kể; không scale ASR riêng được |

---

## 12. Quản lý Session & Connection

### Mô tả

Hai registry toàn cục theo process hoạt động như singleton và có thể truy cập từ bất kỳ đâu trong ứng dụng:

**ConnectionManager** — quản lý vòng đời giao tiếp với client:
Lưu ánh xạ `session_id → WebSocket`. Chịu trách nhiệm gửi mọi loại message về phía client (partial/final transcript, backpressure, lỗi, thông tin session). Là điểm duy nhất để viết ra WebSocket — các layer khác không gọi WebSocket trực tiếp.

**SessionManager** — quản lý trạng thái xử lý của từng client:
Lưu ánh xạ `session_id → session state`. Mỗi session state gồm audio buffer, trạng thái VAD (đang nói/im lặng, bao lâu rồi), trạng thái transcript (partial hiện tại, per-session stabilizer), và hàng đợi inference window. Registry này tạo và xóa session khi client kết nối/ngắt kết nối.

Hai registry được tách riêng có chủ đích: một bên là transport (WebSocket), một bên là domain state (audio, VAD, transcript) — có thể thay đổi cách giao tiếp với client mà không ảnh hưởng logic xử lý, và ngược lại.

Một background task chạy mỗi 60 giây (`_idle_cleanup_loop`) kiểm tra và đóng các session không có hoạt động âm thanh nào trong 5 phút (`_IDLE_SESSION_TIMEOUT_S = 300`).

### Ưu điểm

- **Đơn giản và trực tiếp:** Không cần dependency injection framework; truy cập được từ mọi layer.
- **Tự dọn dẹp:** Idle cleanup tự động tránh memory leak khi client ngắt kết nối bất thường.

### Nhược điểm

- **Không scale ngang:** State nằm trong memory của process — triển khai nhiều instance cần load balancer dùng sticky session để đảm bảo client luôn kết nối đúng server.
- **Khó test isolation:** Global mutable state cần reset thủ công giữa các test case.

### Thay thế đã cân nhắc

| Phương án | Lý do không chọn |
|---|---|
| Redis session store | Cần thiết cho multi-node, nhưng over-engineering cho single-node |
