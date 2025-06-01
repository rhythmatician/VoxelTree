# 🤝 Contributing to VoxelTree

Thanks for your interest in contributing to VoxelTree! This project follows a strict TDD workflow, with automated checks enforced via continuous integration (CI).

---

## ✅ CI Requirements (All Pull Requests)

Every pull request must pass the following before merge:

| Category         | Tool / Requirement                        |
|------------------|--------------------------------------------|
| **Style**        | `black` (code formatter)                  |
| **Lint**         | `flake8` (static analysis)                |
| **Type checks**  | `mypy` (type hints)                       |
| **Tests**        | `pytest` (unit/integration)               |
| **Coverage**     | 90% minimum on touched files              |
| **File format**  | `.npz` patches must include required keys |

---

## 🧪 TDD Commit Structure

Each feature must follow this commit cycle:

1. `RED:` Failing test added
2. `GREEN:` Code added to pass test
3. `REFACTOR:` Code/doc/insight improvements or optimizations

All three commits must be part of a single feature branch, then squashed or merged to `main` when complete.

---

## 📁 Directory Expectations

| Path                  | Description                              |
|-----------------------|------------------------------------------|
| `train/`              | Core model, dataset, training loop       |
| `scripts/`            | Evaluation, ONNX export, generation      |
| `tests/`              | Pytest unit tests                        |
| `docs/`               | Markdown documentation                   |
| `models/`             | Checkpoints + exported ONNX              |

---

## 🚨 Pre-Merge Checks

Your branch must:

- Contain tests for new code
- Include updated `config.yaml` if any config was added
- Not exceed disk usage limits (20GB temp data max)
- Preserve `train.py` CLI compatibility
- Avoid hardcoded paths — use `Path` and `config.yaml`

---

## 📘 Reference Docs

- [📦 Project Outline](docs/PROJECT-OUTLINE.md)
- [🧠 Training Goals](docs/TRAINING-OVERVIEW.md)
- [🤖 Copilot Strategy](.github/copilot-instructions.md)

---

Thanks for helping make terrain beautiful, one LOD at a time!