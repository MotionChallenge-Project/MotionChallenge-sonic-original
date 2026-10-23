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

[Provenance](provenance.json) · [Website](https://motion-challenge-project.github.io/motion-challenge-project/)

Use `--weights DECODER --artifact encoder=ENCODER --offline` for local components.
Use `--termination none` for complete trajectories after a tracking failure.

Validate author inference (C++17 required):
```sh
uv run validate_inference.py . UPSTREAM DECODER ENCODER
```
Use the upstream commit recorded in `provenance.json`.
