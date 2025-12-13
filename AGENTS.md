# AGENTS.md

## Project overview

- Read `docs/code_analysis.md` for the Technical Overview and Architecture Analysis.
- The `docs/.llm_analysis/` directory contains detailed component analyses.
- The current `README.md` is a long AI-generated document, which may not be fully accurate. Refer to `docs/code_analysis.md` for verified information.

## Program targeted use cases

- The program is targeted to run on Mac OS with Apple Silicon chips, and is optimized for local execution of LLMs.
- The program is not designed to be exposed to the public internet or untrusted clients.
  - The clients are trustable localhost or LAN applications that interact with the program via REST API calls.
  - The clients are trusted not to have malicious intent, so security measures against such threats are not a priority.
  - However, the clients may be unreliable and could send malformed or unexpected requests, so robust error handling is necessary.
- The expected cocurreny count is low (typically 1-5 simultaneous requests, 1 is the most common case), so the program is optimized for low-latency single requests rather than high-throughput batch processing.

## Repository expectations

- Keep the `docs/code_analysis.md` and the `docs/.llm_analysis/` directory synchronized with the latest codebase for accurate analyses.

## Resources

- Some core dependency modules can be accessed locally:
  - [mlx-lm](https://github.com/ml-explore/mlx-lm): ~/workspace/custom-builds/mlx-lm
  - [mlx-vlm](https://github.com/zhutao100/mlx-vlm): ~/workspace/custom-builds/mlx-vlm
  - [transformers](https://github.com/huggingface/transformers): ~/workspace/custom-builds/transformers
