# Governance – DOFT Study 01 (Superconductors & Superfluid Helium)

This document describes how the Study 01 repository is maintained and how
decisions are made.

Repository: https://github.com/cesaragostino/doft-study01-superconductors

## 1) Roles

**Maintainer**

- **Cesar Agostino** (GitHub: [@cesaragostino](https://github.com/cesaragostino)) — Repo owner and primary maintainer.

Contact: open a GitHub Issue for general matters. For sensitive topics (e.g.,
potential security issues), use GitHub’s private vulnerability reporting if
enabled, or contact the maintainer directly.

## 2) Scope and decisions

This repository is focused on a single study (“DOFT Study 01”) and its
reproducible pipeline: data, code, figures, and the associated manuscript
draft.

- Day-to-day changes (bug fixes, minor refactors, documentation) are decided by
  the maintainer.
- Substantive changes that alter the main analysis pipeline, data selection, or
  scientific claims should be discussed in Issues before being merged.
- In case of disagreement, the repo owner has final say.

## 3) Branching model

A simple workflow is used:

- Active development may occur on feature branches (`feat/<short-name>`,
  `fix/<short-name>`).
- The `main` branch should remain in a releasable state and reflect the version
  of the code/data used for the current manuscript draft.

## 4) Releases and versioning

When a stable milestone is reached (e.g. a preprint submission), a tagged
release is created:

- Tag format: `vMAJOR.MINOR.PATCH` (SemVer-style).
- Optionally, releases may be mirrored to a DOI-granting archive (e.g. Zenodo).

## 5) Reproducibility and CI

The goal of this repository is to keep the analysis **fully reproducible**:

- The main pipeline should be runnable via a single entry point (e.g.
  `python -m src.run_all`) using the environment specification in
  `environment.yml` or `requirements.txt`.
- Pull requests are encouraged to include any necessary updates to tests,
  notebooks, or figures so that results remain in sync.

Continuous integration (CI) may be added over time to automatically run basic
checks (style, unit tests, and/or a reduced version of the pipeline).

## 6) AI-assisted workflow

AI tools (e.g. OpenAI models) may be used to help with coding, refactoring and
documentation. All AI-assisted changes are reviewed by the maintainer before
being merged. Contributors are asked to mention AI assistance in PR
descriptions when it has significantly influenced the code or text.

## 7) Amending this document

This governance file is intentionally lightweight. Changes can be proposed via
pull request and are accepted once the maintainer agrees and the intent is
clear.
