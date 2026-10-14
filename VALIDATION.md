# Validation of the initial implementation

Validated on 2026-10-13 against Scientific-Workflow fork commit
`a426683b44e3106374f159ecd15d4ca666642cfd`.

| Check | Result |
| --- | --- |
| Workflow regression suite, ordinary and optional external API builds | 201 passed per full run; two existing NPY tests remain ignored. |
| Manager and real PTY tests, Python 3.12 | 13 passed from source and from a standalone installed wheel. |
| Manager tests, Python 3.10 | 10 passed. |
| Native manager unit tests | 3 passed. |
| Format checks and strict native Clippy | Passed. |
| Strict Workflow library Clippy and public API documentation | Passed. |
| Wheel build, independent environment install, CLI success/error paths | Passed; no Rust or Workflow checkout needed at runtime. |
| Real CUDA/MPS, Ledenberg GPU 0 | Passed on RTX 4090, UUID `GPU-0839df87-c919-6855-f429-fc723c417887`. |
| Project test cleanup | 11 captured parent identities and seven MPS endpoints inspected; none remained active. |

The real-GPU acceptance runs verified four simultaneous registered MPS clients,
CUDA kernel results, individual and whole-queue suspension, external STOP/CONT,
safe cancellation of a suspended client, pending queue edits, a second submitting
Python program, ordinary failure isolation, and no automatic job retries.

The GPU-backed terminal run also verified parent-first browsing, actual displayed
PIDs, arrow navigation, job details, logs, holding/releasing/removing/reordering
pending work, viewer detachment and terminal restoration, direct PID jump, retained
finished parents, and persistent dismissal through `t` with artifacts preserved.
This run passed using the independently installed wheel's bundled executable.

A separate shared-endpoint run verified another parent can attach to the existing
MPS service. Cancelling one parent's child preserved the other's client. The owner
refused to shut down a service with attached live clients, stayed controllable and
visible, and completed explicit cleanup after those clients finished.

Generated evidence and terminal screen captures are kept outside both repositories
under the workspace's `.mps-manager/validation/`. These checks establish queue and
MPS behavior; they do not measure performance or validate a scientific model.

The existing ignored Workflow NPY tests require the optional Python 3.14 companion
environment; the new manager has no dependency on that companion or on analysis.
Workflow's pre-existing all-targets Clippy lint under Rust 1.98 was not rewritten;
its library and the new manager's complete targets pass strict Clippy.
