# AGENTS.md

## Docs to read first (by task type)

- Triage / “what is this project?”: `README.md` → `docs/README.md`
- API behavior questions: `docs/apis/` + matching router/service code under `src/mlx_omni_server/`
- Concurrency / responsiveness issues: `docs/concurrency_contract.md` + `src/mlx_omni_server/inference/runtime.py`
- Architecture / roadmap: `docs/code_analysis.md` and `docs/architecture_evaluation.md`
- Active development plans: `docs/dev_plans/`
- Model guidance: `docs/supported_models.md`

## Program constraints

- Target: macOS + Apple Silicon; optimized for local MLX inference.
- Trusted clients on localhost/LAN; not hardened for public internet exposure.
- Expected concurrency is low (typically 1–5); optimize for predictable latency.
- Prefer single-process by default (`--workers 1`) unless you explicitly accept the tradeoffs.

## Conventions and “source of truth”

- **API contracts**: the code + tests are authoritative. Treat docs as summaries that must match implementation.
- **Optional extras** (`images`, `stt`, `tts`): routes stay registered; if deps are missing, return `501` with an install hint (avoid import-time crashes).
  - Prefer runtime gating via `mlx_omni_server.optional_features.ensure_extra_available()` (centralizes `501` errors and install hints).
- **Concurrency contract**: never block the event loop; run MLX-backed work via the shared runtime helpers (see `docs/concurrency_contract.md`).

## Docs maintenance

- Keep `README.md`, `docs/README.md`, and `docs/apis/` consistent with actual behavior.
- Treat `docs/dev_plans/` as forward-looking only; move completed plans to `docs/archive/dev_plans/`.
- If you change behavior that was previously documented as a “gap” or “issue”, remove/adjust it in docs to avoid stale TODO lists.

## Workflows and commands

- Packaging note: this repo is **not published to PyPI**; `pip install mlx-omni-server` installs a different project. Use source installs (`pip install -e ".[...]"`) or `uv sync`.
- Common local dev commands (install/run/format/test): `docs/development_guide.md`.
- `tests/integration/` may download models and can be slow/heavy.

## Local-only developer config

Use `config/local-resources.yaml` (gitignored; template: `config/local-resources.example.yaml`) to record machine-local resources and paths to external reference repos used during implementation/research, so we don’t hardcode local filesystem paths into docs or patches.

- Create it via: `cp config/local-resources.example.yaml config/local-resources.yaml`
- Do not guess external repo locations. Always resolve via `config/local-resources.yaml`; if missing, ask the operator to create it from the example template.
- If you use `pyenv`, record the env name under `local_envs.pyenv_env_name` (avoid hardcoding it in docs/tests).
- Before finalizing a patch, ensure no machine-local paths leak into tracked files (prefer `$HOME/...` or relative paths in docs/tests). Quick check: `rg -n "/Users/" .`
