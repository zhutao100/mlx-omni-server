# Architecture Evaluation and Improvement Plan

## Scope (targeted use cases)

This plan is scoped to the project’s stated goals (see `docs/stack_assessment.md`):

- **macOS + Apple Silicon** local inference, optimized for MLX and unified memory.
- **Localhost/LAN, trusted clients** (not hardened for the public internet).
- **Low concurrency (typically 1–5)**, optimized for low-latency single requests over high throughput.
- **Robustness to malformed/unexpected requests** matters more than adversarial security.

Derived architectural principles:

- Keep the deployment **single-process by default** (`workers=1`) and make contention explicit.
- Prefer **predictable latency** over “best effort” parallelism that risks unified-memory spikes.
- Make optional modality backends **pluggable and isolatable** (extras + clear interfaces).
- Treat “unreliable clients” as an input-quality problem: **bounded resources + good errors**.

---

## Current architecture (spot-checked against implementation)

### Overall assessment

**Clearness: Good (7/10).**
The project reads like a conventional FastAPI “modular routers + service layer” design. The separation by capability (chat/embeddings/images/stt/tts/responses) and OpenAI-compatible schemas are clear and make the codebase approachable.

**Health: Good (7/10).**
The server now enforces a consistent operational contract across endpoints: “never block the event loop; gate MLX-backed compute; ensure request-scoped filesystem artifacts are unique.” The chat/responses path remains the most mature, but embeddings/images/STT/TTS now follow the same execution model.

**Robustness: Good, with remaining operational risks (7/10).**
The baseline runtime contract is in place (threadpool offload + shared MLX gate + request-scoped artifacts). Remaining risks are primarily about **bounded execution/backpressure**, **memory-aware lifecycle management**, **multi-worker safety**, and the **dependency/plugin surface** (audio/image stacks).

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

## Key remaining architectural gaps (priority order)

### 1. Backpressure is missing (unbounded waiting behind the MLX gate)

The shared “one-at-a-time” MLX gate is stable for Apple unified memory, but without a bounded queue or explicit rejection policy, clients can pile up and experience very high tail latency.

For your low-concurrency target, the goal is not high throughput; it is to make overload behavior predictable:

- bounded waiting (queue depth)
- explicit 429/503 behavior
- clear client guidance (“retry after”, etc.)

### 2. Model lifecycle + caching is still fragmented (no budgets/admission control)

Chat/embeddings/images have explicit in-process caches; STT/TTS mostly rely on underlying libraries. There is no single place to:

- enforce memory budgets / headroom
- evict models consistently (LRU/TTL)
- prewarm hot models safely
- observe cache hit/miss and model load time

### 3. Multi-worker configuration is an attractive footgun

`uvicorn --workers N` creates **N processes**, each with its own caches and its own “global” locks. If the CLI exposes `workers`, it is easy to configure into GPU overcommit and fail unpredictably unless explicitly constrained.

### 4. Dependency surface: keep optional modalities truly optional

This codebase already isolates the long-tail modality stacks behind install extras (images/STT/TTS). The main remaining work here is **discipline**, not a refactor:

- avoid pulling optional backends into the core dependency set,
- keep imports safe at module import time (routes should stay registered and return `501` with an install hint),
- keep each modality’s “extra params” surface documented and tested.

See `docs/stack_assessment.md` for the constraint-driven rationale.

### 5. “Unreliable clients” hardening is incomplete (limits + error shape)

FastAPI/Pydantic give good schema validation, but robustness also needs explicit operational limits and consistent error responses:

- upload / base64 payload size limits (413 instead of a 500)
- timeouts on long-running requests
- consistent OpenAI-style error payloads across endpoints (including overload/backpressure)
- logging defaults are now capped and skip binary/streaming bodies; remaining gaps are redaction/sampling and configurable exclusions

### 6. Observability is not yet tied to the runtime contract

Given the MLX gate is a central contention point, it should be measurable: queue length, time waiting, execution time, and cancellation rate.

---

## Baseline (implemented)

- Runtime/behavioral contract: `docs/concurrency_contract.md`
- Archived Phase 0 checklist: `docs/archive/architecture_phase0_baseline.md`

---

## Architecture improvement plan

### Phase 1: make overload behavior explicit (bounded runtime + metrics)

Evolve the existing runtime (`src/mlx_omni_server/inference/runtime.py`) into a small “inference runtime” layer used by all services.

**Responsibilities:**

1. **Bounded execution**

   * Keep MLX gating, but add a bounded wait/queue policy so overload doesn’t silently turn into minutes of tail latency.

2. **Resource gating / scheduling**

   * Start conservative (`k=1`) but support explicit “fast lane vs heavy lane” options when needed (e.g., embeddings vs diffusion).
   * Define and implement backpressure: reject with 429/503 once queue is full.

3. **Observability hooks**

   * Track at minimum: queue length, time waiting for gate, execution time, cancellation count, model load time, cache hit/miss.

4. **Uniform cancellation semantics**

   * If clients disconnect, stop streaming immediately; optionally cancel queued work if not started.

**Resulting endpoint flow:**

* Router validates request → Service builds a “job” → Runtime executes job with gating + threadpool → Service formats response.

This makes “robustness” a shared property, not something each component reimplements (or forgets).

---

### Phase 2: centralize lifecycle + budgets (predictable unified-memory behavior)

Build a small lifecycle manager that sits above modality-specific services and makes “what is loaded” and “what can run” a deliberate policy.

1. **Per-capability / per-model semaphores**

   * Only if Phase 1 metrics show it’s valuable: allow limited parallelism for lighter jobs (e.g., embeddings) without destabilizing diffusion/chat workloads.

2. **Memory-aware admission control**

   * Track approximate peak memory per job type/model (even coarse heuristics help).
   * Reject, queue, or downshift concurrency when predicted peak exceeds configured headroom.

3. **Unified caching + eviction**

   * Standardize how models/generators are cached and evicted across chat/embeddings/images (and any explicit STT/TTS caching you add).

4. **Multi-worker safety**

   * Default workers=1 for MLX-bound workloads unless you implement cross-process coordination.
   * If you keep `workers` configurable, enforce safety: warn loudly, or hard-block >1 unless `--allow-unsafe-workers` is set.

---

### Phase 3: dependency isolation + maintainability (from stack assessment)

This phase is about reducing churn from the modality ecosystem without changing the core architecture.

1. **Split optional capabilities into install extras**

   * Example: split into extras like `[chat]`, `[embeddings]`, `[images]`, `[stt]`, `[tts]` (installable from source via `pip install -e ".[...]"`).
   * Code should degrade gracefully when a backend isn’t installed (e.g., return a clear 501/400 with guidance).

2. **Keep modality backends behind narrow interfaces**

   * The TTS adapter pattern is a good precedent; apply the same pattern where it helps (STT/image backends, optional “second runtime lane”).

3. **Hardening & testing**

   * Add a smoke-test matrix per extra/backend (pin upgrades are coordinated).
   * Add concurrency/load tests that cover:

     * event-loop blocking under concurrent load
     * concurrent TTS requests (no cross-request artifact collisions)
     * concurrent image generations (artifact uniqueness + cleanup)
     * mixed-load unified-memory spikes / OOM scenarios
   * Add soak tests with cancellation/disconnect behaviors (especially for streaming).

4. **Operational polish**

   * Request limits (upload sizes, max base64 sizes), timeouts, and consistent OpenAI-style error payloads.
   * Structured logging + request IDs (request IDs exist; body logging is capped and skips binary/streaming by default; remaining: redaction/sampling/exclusions).
   * Metrics/Tracing (minimal counters are enough): latencies by endpoint/model, queue wait time, model load time, error rates.

---

## Practical “definition of done” checkpoints

* **DoD-1:** Overload behavior is explicit: bounded queues + 429/503 (no unbounded waiting).
* **DoD-2:** Running with `--workers > 1` is either safe-by-design or explicitly prevented.
* **DoD-3:** You can explain (and measure) the scheduling policy: “what runs concurrently, why, and with what limits.”
