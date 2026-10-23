# SONIC v1.1

Independent Motion Challenge adapter for [SONIC v1.1](https://github.com/NVlabs/GR00T-WholeBodyControl).

```sh
git clone --recurse-submodules https://github.com/motion-challenge-project/motion-challenge-sonic-v1-1.git
cd motion-challenge-sonic-v1-1
uv sync
uv run evaluate.py
```

The root `motion-challenge` submodule owns datasets, physics, failure rules and scoring.
Local weights are reused; downloads are hash-checked. Results are stored outside Git.
Use `--weights PATH --data PATH --offline` for cached artifacts.

[Provenance](provenance.json)

Use `--weights DECODER --artifact encoder=ENCODER --offline` for local components.
Use `--termination none` for complete trajectories after a tracking failure.

Validate author inference (C++17 required):
```sh
uv run validate_inference.py . UPSTREAM DECODER ENCODER
```
Use the upstream commit recorded in `provenance.json`.

## Checkpoint source

[NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/45c03facbf848e0c743091b7d9c17952096ff908/README.md) · [model_decoder.onnx](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/sonic_v1_1/model_decoder.onnx). Author-released checkpoint.

The adapter pins this published artifact by SHA256. Artifact identity is separate from inference parity.

One `uv run evaluate.py` produces one verified `result.json`: success, completion,
three MPBPE alignments through first failure and the same three over the full motion.
Both scopes use the same checkpoint and source revision; no branch switching is needed.
