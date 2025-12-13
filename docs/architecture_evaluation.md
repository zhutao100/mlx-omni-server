## Architecture evaluation (based on `docs/` and spot-checking the current implementation)

### Overall assessment

**Clearness: Good (7/10).**
The project reads like a conventional FastAPI “modular routers + service layer” design. The separation by capability (chat/embeddings/images/stt/tts/responses) and OpenAI-compatible schemas are clear and make the codebase approachable.

**Health: Mixed (5/10).**
The chat/responses path is mature and production-oriented (caching, stream multiplexing, cancellation, cleanup). However, the other capability paths (embeddings/images/stt/tts) do not consistently follow the same operational contract: “never block the event loop; gate MLX-backed compute; ensure request-scoped filesystem artifacts are unique.”

**Robustness: Uneven, with critical concurrency hazards (4/10).**
The server can be robust under concurrent load only insofar as clients predominantly hit the chat/responses endpoints. Under mixed workloads, embeddings/images/STT/TTS can degrade the entire process (event-loop blocking), images can trigger GPU contention / OOM, and multiple endpoints have request-safety hazards (e.g., shared filenames).

---

## What is strong / healthy

1. **OpenAI-compatibility as a first-class product decision**

   * Schemas and routes aligned with common clients reduces integration friction.

2. **Chat architecture is notably strong**

   * Correct use of threadpool for blocking work, plus a global serialization mechanism.
   * Request caching + streaming multiplexing + cancellation when all clients disconnect are strong operational features.

3. **Modularity is sensible**

   * Routers aggregated centrally; services encapsulate model/library specifics.
   * Responses endpoint is a clean adapter layered over chat (minimal duplication, inherits chat’s robustness).

4. **Evidence of maturity**

   * Comprehensive tests and examples; suggests active maintenance and user focus.

---

## Primary architectural risks (in priority order)

### 1. Inconsistent concurrency contract across components (systemic)

You effectively have **two different servers** in one process:

* Chat/responses: “async-safe, MLX-gated, threadpooled”
* Embeddings/Images/STT/TTS: “sync calls inside async + no gating” (and there are shared/collision-prone filesystem artifacts)

This is the single biggest threat to reliability and predictability.

### 2. Event loop blocking (embeddings/images/STT/TTS) can freeze *all* endpoints

A single long embeddings/images/STT/TTS call can block the reactor and stall unrelated traffic (including health checks). This is a production-stopper. STT/TTS and image generation are the most likely to be long-running, but embeddings can also become expensive with large/slow models.

### 3. GPU/MLX contention policy is implicit and not unified

Chat uses a global `mlx_lock` (serialize all heavy work). Others do not. Even if MLX/mflux/whisper were thread-safe, **memory pressure is not**, and Apple unified memory makes contention especially easy to trigger.

### 4. Multi-worker configuration is likely unsafe by default

`uvicorn --workers N` creates **N processes**, each with its own caches and its own “global” locks. If the CLI exposes `workers`, it is easy to configure into GPU overcommit and fail unpredictably unless explicitly constrained.

### 5. Model lifecycle / caching is fragmented

Each capability relies on its library’s implicit caching or ad-hoc caches. There is no single place to:

* enforce memory budgets
* evict/TTL models consistently
* observe cache hit/miss
* prewarm hot models safely

Additionally, some caches are currently scoped too narrowly to help across requests (e.g., images generator caching on a per-request service instance), which makes performance and memory behavior less predictable.

---

## Supporting evidence (from current code)

These issues are directly visible in the current implementation:

* Chat MLX gate + threadpool: `src/mlx_omni_server/chat/generation_service.py:95`
* Embeddings runs sync inference inside an `async` route: `src/mlx_omni_server/embeddings/router.py:19`
* Images instantiates its service per request and runs sync generation inside an `async` route: `src/mlx_omni_server/images/images.py:21`
* Images output IDs are second-based (`int(time.time())`), so they can collide across concurrent requests: `src/mlx_omni_server/images/images_service.py:186`
* STT performs synchronous transcription within an async request path: `src/mlx_omni_server/stt/whisper_model.py:129`
* TTS uses a shared output path (`sample.wav`): `src/mlx_omni_server/tts/tts_service.py:87`
* Multi-worker mode is exposed (`--workers`) and maps to `uvicorn workers=`: `src/mlx_omni_server/main.py:55`

---

## Architecture improvement plan

### Phase 0 (same-day fixes): stop correctness bugs and server-freeze paths

1. **Eliminate event-loop blocking for embeddings/images/STT/TTS immediately**

   * Wrap *all* blocking inference/transcription/generation calls with `run_in_threadpool` (or `anyio.to_thread.run_sync`) from the async endpoints.
   * Ensure cancellation propagates (if client disconnects, stop work where possible; otherwise stop streaming and release resources).

2. **Fix TTS file race condition**

   * Replace hardcoded `sample.wav` with request-scoped temp files (`tempfile.NamedTemporaryFile` / `TemporaryDirectory`) or fully in-memory buffers when feasible.
   * Make cleanup exception-safe.

3. **Fix image artifact collisions and clarify artifact lifecycle**

   * Use request-scoped unique names (UUIDs) for any on-disk artifacts.
   * Decide whether URL-mode images are long-lived cache artifacts or truly temporary files, and document/implement cleanup accordingly.

4. **Introduce a shared MLX gating mechanism used everywhere**

   * Minimum viable: reuse the same global `mlx_lock` (or a shared `asyncio.Semaphore(1)`) for **chat, embeddings, images, stt, tts**.
   * This is conservative but immediately stabilizes the system under concurrent load.

Deliverable: `docs/concurrency_contract.md` (1–2 pages) stating:

* “No blocking on event loop”
* “All MLX work goes through the gate”
* “All request-scoped filesystem artifacts are unique”

---

### Phase 1 (1–2 weeks): unify the runtime model with an explicit “Inference Runtime” layer

Create a new internal subsystem (name suggestion: `inference_runtime/`) used by all services.

**Responsibilities:**

1. **Execution offload**

   * A single place that runs blocking work in a bounded thread pool.

2. **Resource gating / scheduling**

   * Start with `Semaphore(k)` where `k=1` (conservative).
   * Add structured queueing and backpressure (reject or 503/429 when queue is full).

3. **Observability hooks**

   * Track: queue length, time waiting for gate, execution time, memory errors, model load time, cache hit/miss.

4. **Uniform cancellation semantics**

   * If clients disconnect, stop streaming immediately; optionally cancel queued work if not started.

**Resulting endpoint flow:**

* Router validates request → Service builds a “job” → Runtime executes job with gating + threadpool → Service formats response.

This makes “robustness” a shared property, not something each component reimplements (or forgets).

---

### Phase 2 (2–6 weeks): make concurrency policy smart instead of purely serialized

The global “one-at-a-time” gate is stable but can leave performance on the table. Replace it with an explicit policy:

1. **Per-capability / per-model semaphores**

   * Example: allow 2 concurrent embeddings jobs but only 1 diffusion job.
   * Or per-model weights: diffusion consumes more “capacity units” than embeddings.

2. **Memory-aware admission control**

   * Track approximate peak memory per job type/model (even coarse heuristics help).
   * Reject or queue requests when predicted peak exceeds a configured headroom threshold.

3. **Separate “fast lane” vs “heavy lane”**

   * Avoid embeddings being stuck behind long image generations if you want snappy UX.

4. **Multi-worker safety**

   * Default workers=1 for MLX-bound workloads unless you implement cross-process coordination.
   * If you keep `workers` configurable, enforce safety: warn loudly, or hard-block >1 unless `--allow-unsafe-workers` is set.

---

### Phase 3 (ongoing): lifecycle, consistency, and maintainability

1. **Centralize model lifecycle management**

   * Standardize: load, cache, evict, warmup, and “model handle” ownership.
   * Implement LRU/TTL with explicit memory budgeting where possible.

2. **Normalize service instantiation patterns**

   * Prefer shared service instances (as chat/embeddings do) unless there is a strong reason not to.
   * If per-request services exist, ensure they are stateless wrappers around shared caches/runtimes.

3. **Hardening & testing**

   * Add concurrency/load tests that reproduce:

     * STT/TTS event loop freeze regression
     * TTS file collision regression
     * images filename collision regression
     * images + chat mixed-load OOM regression
   * Add soak tests with cancellation/disconnect behaviors (especially for streaming).

4. **Operational polish**

   * Structured logging + request IDs.
   * Metrics/Tracing (OpenTelemetry or Prometheus-style counters): latencies by endpoint/model, queue wait time, model load time, error rates.

---

## Practical “definition of done” checkpoints

* **DoD-1:** No endpoint calls a blocking ML library function on the event loop.
* **DoD-2:** All MLX-backed compute goes through a shared gate (initially serialized).
* **DoD-3:** TTS produces correct outputs under concurrent requests (no shared filenames, no cross-request artifact collisions).
* **DoD-4:** Running with `--workers > 1` is either safe-by-design or explicitly prevented.
* **DoD-5:** You can explain (and measure) the scheduling policy: “what runs concurrently, why, and with what limits.”

---

If you want, I can also draft the concrete module/API shape for the proposed `inference_runtime` (interfaces, suggested classes, where the gate lives, how jobs are represented, and how to retrofit each component with minimal diff).
