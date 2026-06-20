# Design Decisions

This document explains the rationale behind each major architectural choice, including trade-offs and alternatives considered.

---

## 1. Audio Buffer — Ring Buffer

### Description

Incoming audio from the client is stored in a fixed-capacity **ring buffer** (default 12 seconds). When full, the oldest samples are overwritten rather than allocating more memory. Every inference window is read from this buffer as a snapshot.

### Pros

- **Fixed, predictable memory:** No dynamic allocation at runtime — no GC pauses under high load.
- **Flexible reads:** Any time range within the buffer can be extracted (latest N seconds, or an arbitrary slice) without redundant copies.

### Cons

- **Fixed capacity:** If an inference window is larger than the buffer capacity, it will be silently truncated.
- **Single-writer assumption:** No synchronization mechanism — safe with one receive loop, but needs revisiting if additional writers are introduced.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Queue of individual audio packets | Cannot query by time range; higher memory overhead |
| Write to temp file, read back | High latency; unnecessary complexity |

---

## 2. VAD Model — Silero VAD

### Description

The system uses **Silero VAD** — a compact deep learning model exported to ONNX, running via ONNX Runtime with no PyTorch dependency. The model analyzes ~32ms frames and returns **per-frame speech probabilities**, not just a binary decision.

### Pros

- **Per-frame probabilities:** A hard requirement for the Speech Trimmer — it needs a probability map to precisely locate speech start/end within the window, not just a yes/no answer.
- **No PyTorch at runtime:** Runs via ONNX Runtime — significantly reduces Docker image size and startup time compared to loading full PyTorch.
- **High accuracy:** Outperforms rule-based approaches across diverse audio environments, especially with accented speech and mild background noise.
- **INT8 quantization support:** Can quantize on first startup, reducing RAM ~50% and speeding up CPU inference.
- **Small model size:** ~1–2 MB — baked into the Docker image with negligible size impact.

### Cons

- **Higher latency than rule-based:** Neural network inference is slower than WebRTC VAD, though mitigated by the VAD pool and ONNX runtime.
- **Black box:** Cannot explain why a specific frame was classified as silence — difficult to debug on unusual speech patterns.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| WebRTC VAD (py-webrtcvad) | Rule-based, returns binary only — no per-frame probabilities, Speech Trimmer cannot use it |
| PyAnnote Audio VAD | High accuracy but heavy PyTorch dependency; unnecessary when Silero is sufficient |
| Whisper VAD | Built into Whisper but not separable; too heavy for a pre-filter step |
| Energy-based VAD (RMS threshold) | Simplest, but inaccurate under background noise; returns no per-frame probabilities |

---

## 3. VAD Pool

### Description

Instead of sharing a single VAD model across all sessions, the system pre-initializes a **pool** of multiple VAD instances (default 8). When speech detection is needed, an instance is checked out from the pool, runs on a dedicated thread, and is returned immediately after.

### Pros

- **True parallelism:** Multiple sessions can run VAD simultaneously without contention.
- **Reuse of frame probabilities:** VAD probabilities computed for the speech decision are reused directly by the Speech Trimmer — no second ONNX pass required.
- **Backpressure on exhaustion:** When all instances are busy, the system sends a `backpressure` signal rather than queuing indefinitely or blocking the receive loop.

### Cons

- **Increased startup time:** Loading multiple VAD instances in parallel on server startup adds a few seconds to readiness.
- **Pool exhaustion:** When all instances are occupied, new inference windows are dropped and backpressure is sent to the client.
- **Linear memory with pool size:** Each instance consumes ~10–20 MB RAM; a pool of 8 uses 80–160 MB for VAD alone.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Single shared instance + lock | Severe bottleneck under concurrent sessions |
| PyTorch SileroVAD | Heavy dependency; unnecessary |
| WebRTC VAD | Less accurate; no per-frame probabilities |

---

## 4. VAD Trigger Strategies

### Description

VAD returns per-frame speech probabilities (~32ms frames). A **trigger strategy** converts this probability sequence into a binary speech/silence decision for the window. Three strategies are available:

- **Consecutive frames:** Triggers only when N consecutive frames all exceed the threshold.
- **EMA smoothed:** Smooths probabilities via exponential moving average before threshold comparison — reduces false positives from short noise bursts.
- **State machine** *(default via settings.yaml)*: FSM with a higher onset threshold than offset threshold — hysteresis prevents state chattering when probabilities hover near the threshold.

### Pros

- **Configurable without code changes** — suitable for different deployment environments and noise profiles.
- **EMA provides a good default balance** between sensitivity and noise rejection.
- **State machine handles noisy environments well** — hysteresis prevents rapid state toggling.

### Cons

- **Consecutive frames** is sensitive to impulse noise (clicks, pops) if N is too low.
- **EMA** introduces a small lag at utterance start and end due to moving average inertia.
- **State machine** requires tuning two separate thresholds — harder to configure correctly from scratch.

---

## 5. Adaptive Inference Pacing

### Description

Instead of running inference at a fixed interval, the system dynamically adjusts the pacing per session:

- **`ONSET_INTERVAL_MS` (400ms)** — used right after new speech is detected; favors fast partial updates.
- **`STABLE_INTERVAL_MS` (1200ms)** — used when the transcript has not changed across consecutive windows; reduces redundant ASR calls.

The interval resets to `ONSET_INTERVAL_MS` whenever `vad_state.last_speech_time` advances (new speech frames detected).

### Pros

- **Reduces redundant ASR calls** during stable speech — fewer GPU calls, lower NeMo load.
- **Fast partials when it matters** — the onset window keeps latency low at the start of each utterance.
- **Per-session granularity** — each session tracks its own `current_interval_ms`, so sessions do not interfere with each other's pacing.

### Cons

- **Two parameters to tune** instead of one (`ONSET_INTERVAL_MS` and `STABLE_INTERVAL_MS`), which may be confusing initially.
- **Stability detection is simplistic** — compares current stabilized output against `last_partial_for_stability`; does not account for minor changes caused by trailing words.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Fixed interval only | Wastes ASR calls on stable speech; or if interval is long, misses fast partial updates |
| Backoff based on ASR queue depth | More complex; queue depth is a lagging signal compared to transcript stability |

---

## 6. Trailing-Silence Window Correction

### Description

A stale VAD problem occurs at the end of an utterance: the speaker has stopped, but the 6-second inference window still contains old speech frames from earlier, so `is_speech` returns `True`. This triggers unnecessary ASR calls on pure-silence windows.

The correction overrides `is_speech=True` to `False` when the last speech segment in the current window ended more than `TRAILING_SILENCE_MS` (1000ms) ago — before `VADState.update()` is called.

### Pros

- **~50% fewer ASR calls at utterance end** — the most common place for stale VAD decisions.
- **Zero additional ONNX inference** — the correction is a pure Python check on `vad_state.last_speech_time`, reusing data already computed.

### Cons

- **May suppress ASR on very short pauses** if `TRAILING_SILENCE_MS` is set too low — words separated by a natural pause mid-sentence could be cut off.
- **Depends on accurate `last_speech_time`** — if VAD itself misreads the last speech frame, the correction boundary shifts accordingly.

---

## 7. Speech Trimming + MIN_TRIMMED_AUDIO_MS Gate

### Description

Before calling ASR, the 6-second inference window is trimmed to the detected speech region using per-frame VAD probabilities (already computed, no second ONNX pass). The trimmed audio has `SPEECH_PADDING_MS` (200ms) added on each side to preserve context.

After trimming, a length gate checks: if the result is shorter than `MIN_TRIMMED_AUDIO_MS` (500ms), the window is skipped entirely without calling ASR.

### Pros

- **Reduces ASR input size** — shorter audio → faster NeMo inference and lower network overhead.
- **Reuses VAD probs** — `segments_from_probs()` is pure Python; no second ONNX pass needed.
- **MIN_TRIMMED_AUDIO_MS gate prevents near-silent ASR calls** — avoids sending windows where VAD detected only a tiny blip of probable speech.

### Cons

- **Trimming can fail** if VAD probabilities are noisy — the fallback is to use the full window, which is safe but negates the trimming benefit.
- **MIN_TRIMMED_AUDIO_MS is a hard threshold** — a genuine short word like "ừ" (Vietnamese for "yes") close to the 500ms boundary may be skipped.

---

## 8. Global ASR Semaphore

### Description

A single `asyncio.Semaphore` (`ASR_SEMAPHORE_LIMIT`, default 8) caps the total number of concurrent NeMo HTTP requests across **all sessions**. Each inference worker must acquire the semaphore before calling ASR.

### Pros

- **Protects NeMo from overload** — prevents a sudden spike in sessions from flooding the ASR server with concurrent requests.
- **Shared cap is more efficient** than per-session limits — a session that is idle or slow doesn't waste quota, and a busy session can use more.

### Cons

- **Global contention under high load** — sessions compete for the same semaphore; a session processing a long audio window can delay others.
- **No priority** — all sessions are equal; there is no mechanism to prioritize newer sessions or those closer to finalization.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Per-session semaphore | Inefficient — idle sessions hold quota; no protection against coordinated spikes |
| No semaphore (unlimited concurrency) | NeMo server would be overloaded under high session count |

---

## 9. Stabilization Pipeline

### Description

Streaming ASR continuously produces new hypotheses — and new hypotheses sometimes shorten or diverge from the previous one, causing **rollback** (displayed text visibly shrinks). The stabilization pipeline has two layers:

**Layer 1 — Longest Common Prefix (LCP):**
Compares the new hypothesis against the previous one and keeps only the common prefix. The unstable tail is held back.

| Mode | Mechanism | When to use |
|---|---|---|
| **Word-level** *(default)* | Compares word-by-word (split on whitespace). Example: `"xin chào bạn"` vs `"xin chào anh"` → common prefix: `"xin chào"` | Vietnamese and space-delimited languages |
| **Character-level** | Compares character-by-character. Example: `"xin chào bạn"` vs `"xin chào anh"` → common prefix: `"xin chào "` | Languages without spaces (Chinese, Japanese) or when finer granularity is needed |

Word-level is safer — it never cuts mid-word, avoiding the display of partial tokens that immediately disappear.

**Layer 2 — Rollback Suppression:**
Protects already-displayed text from being shortened. Multiple strategies are available:

| Strategy | Mechanism | When to use |
|---|---|---|
| **Frozen prefix** *(default)* | Progressively freezes the prefix after N consistent hypotheses; rejects any hypothesis that contradicts the frozen region | Good general balance for Vietnamese |
| **Hard length** | Word count can only increase, never decrease | When downstream consumers cannot accept deletions |
| **Edit distance** | Rejects hypotheses whose word edit distance from last output exceeds a threshold | High-noise environments |
| **N-consecutive** | Only commits a rollback after N consecutive frames all agree on the shorter hypothesis | When correctness is more important than latency |

### Pros

- **Per-session state:** Each session has its own stabilizer instance, reset after each `finalize()` — no bleed-over between sessions or utterances.
- **Modular:** Strategies are independent and can be chained via `StabilizerPipeline`.
- **Vietnamese-friendly:** Word-level LCP is well suited to Vietnamese tokenization (space-delimited).

### Cons

- **Adds display latency:** An overly conservative freeze threshold means text appears later than the ASR actually produces it.
- **Cannot self-correct severe ASR errors:** Some strategies (e.g. `hard_length`) lock incorrect state and cannot recover without a session reset.

---

## 10. Per-Session Inference Worker

### Description

Each WebSocket session gets its own **dedicated background worker** (`asyncio.Task`) that drains the inference queue. The audio receive loop only enqueues windows — it never waits for VAD or ASR to complete.

### Pros

- **Receive loop is never blocked:** Even if ASR takes 2–3 seconds, the client continues streaming and the buffer keeps updating normally.
- **Explicit backpressure:** When the queue is full, a `backpressure` signal is sent to the client rather than silently dropping or blocking.
- **Session isolation:** A timeout or error in one session does not affect others.

### Cons

- **Queue drift:** If ASR is slower than the rate at which windows are produced, the queue accumulates stale windows — transcripts may lag behind actual speech.
- **No priority queue:** Windows are processed FIFO; there is no mechanism to prefer the most recent window when congested.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Inline processing in receive loop | Completely blocks audio reception when ASR is slow |
| Dedicated thread per session | Higher overhead; unnecessary with async I/O |
| Dedicated process per session | Excessive overhead; high IPC latency |

---

## 11. ASR Engine — External HTTP Service (NVIDIA NeMo)

### Description

ASR runs as a **separate external service**. The streaming service communicates with the NeMo server over HTTP, sends audio as a WAV file, and receives the transcript string back. HTTP connections are reused via connection pooling (`aiohttp.ClientSession`).

### Pros

- **Full decoupling:** The ASR engine can be replaced (Whisper, Google STT, etc.) without touching the streaming service.
- **Independent scaling:** The NeMo server can be scaled separately (GPU count, replicas) according to ASR demand, independent of the streaming service.
- **Hard timeouts:** Each request has explicit timeouts (`ASR_CONNECT_TIMEOUT`, `ASR_REQUEST_TIMEOUT`) — a slow ASR call is cancelled rather than hanging a session indefinitely.

### Cons

- **Network dependency:** If the NeMo server is down or the network is slow, the entire ASR pipeline stops. No fallback or circuit breaker is implemented.
- **No retry:** Failed requests are silently dropped — no retry mechanism.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| gRPC streaming | Lower latency, but significantly more complex while HTTP API is sufficient |
| Embed model directly in service | Large RAM increase; cannot scale ASR independently |

---

## 12. Session & Connection Management

### Description

Two global, process-level registries serve as singletons accessible from anywhere in the application:

**ConnectionManager** — manages the communication lifecycle with clients:
Holds the `session_id → WebSocket` mapping. Responsible for sending all message types to the client (partial/final transcripts, backpressure, errors, session info). The single point of writing to WebSocket — other layers never call WebSocket directly.

**SessionManager** — manages the processing state of each client:
Holds the `session_id → session state` mapping. Each session state contains the audio buffer, VAD state (speaking/silence, duration), transcript state (current partial, per-session stabilizer), and the inference queue. This registry creates and removes sessions on connect/disconnect.

The two registries are intentionally separated: one owns transport (WebSocket), the other owns domain state (audio, VAD, transcript). This allows the communication layer to change independently of the processing logic, and vice versa.

A background task runs every 60 seconds (`_idle_cleanup_loop`) and closes sessions with no audio activity for 5 minutes (`_IDLE_SESSION_TIMEOUT_S = 300`).

### Pros

- **Simple and direct:** No dependency injection framework needed; accessible from any layer.
- **Self-cleaning:** Idle cleanup automatically prevents memory leaks when clients disconnect abnormally.

### Cons

- **Does not scale horizontally:** State lives in process memory — multi-instance deployments require sticky session routing at the load balancer.
- **Difficult to test in isolation:** Global mutable state requires manual reset between test cases.

### Alternatives considered

| Option | Reason not chosen |
|---|---|
| Redis session store | Necessary for multi-node, but over-engineering for single-node deployment |
