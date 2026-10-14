# Development boundaries

`mps-manager` and the Scientific-Workflow fork have independent Git histories and
release artifacts. The dependency flows from this repository's native crate to
the fork's `external-tools` API. Changes to MPS protocol, queue commands, Python
ergonomics, and the queue browser belong here. General changes to Workflow's
expander, owned process groups, and shared terminal components belong in the
fork; upstream reviews can therefore stay small and independent of MPS.

The pinned Git dependency lives in `native/Cargo.toml`, and its tested fork
revision is recorded in `WORKFLOW_DEPENDENCY.md`. Cargo fetches that revision
directly, so development and source installs require only this repository.
Update the manifest, lockfile, and recorded revision together. Built wheels bundle this repository's
compiled native executable; users need only Python and the NVIDIA tools for MPS.

All jobs are independent script processes. Keep arguments explicit; do not
interpret scientific state, construct checkpoints, derive progress/ETA, inject
seeds, or add automatic retries. Generated control files stay under the hidden
state directory. Same-host connection must verify parent identity, and MPS
cleanup must finish before killing client processes or dismissing a parent.

CPU integration tests exercise real parent/child processes. PTY tests inspect
rendered screens and verify terminal restoration. Real-GPU acceptance tests are
separate scripts requiring an explicitly assigned GPU; they are not run in CI.

## Build and test

```bash
cargo build --locked --manifest-path native/Cargo.toml
export MPS_MANAGER_BINARY="$PWD/native/target/debug/mps-manager-native"
export PYTHONPATH="$PWD/src"
python -m pip install '.[test]'
python -m unittest discover -s tests -p 'test_*.py' -v
cargo test --locked --manifest-path native/Cargo.toml
cargo clippy --locked --manifest-path native/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path native/Cargo.toml -- --check
```

Build release wheels with `python -m pip wheel . --no-deps -w dist`.
The packaged binary runs independently of the source checkout.

The example GitHub Actions pipeline is preserved in `.github/ci-template.yml`.
Move it to `.github/workflows/ci.yml` when enabling Actions with credentials that
allow workflow updates. It builds this repository directly and fetches the pinned
Workflow dependency through Cargo.
