# Validation of the initial implementation

## Version 0.2.0: shared services and fixed arguments

Validated on 2026-10-15 with 17 Python integration/PTY tests, three native unit
tests, strict Clippy, and formatting checks. The same tests pass against the
installed wheel in base. Installation uses `--no-deps`; the existing OpenMM,
Polychrom, NumPy, Torch, and Triton installations are reused.

A bounded real-GPU run verified automatic sharing on GPU 0, inherited GPU-wide
limits, round-robin admission across parents, FIFO within each parent, retained
slots during suspension, common arguments reaching actual CUDA kernels, and
fixed/resolved parameter metadata. The TUI displayed both parents independently.
Terminating the first parent through `t` preserved the second parent's CUDA job;
the final parent closed the owned service. A second phase ran CUDA jobs from two
parents concurrently in one service. Captured admission order was B0, A0, B1, A1,
B2, A2 after an earlier A job released its slot.

The prior v0.1 results below describe that release's original explicit-attachment
behavior; automatic service sharing supersedes its first-parent ownership rule.

## Version 0.1.0

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

## Pause/resume shortcuts (0.2.1)

The visible `p` and `r` controls pause/resume the selected queue on the parents
screen and the selected child in its queue, details, or logs. Real PTY checks
verify queue and child suspension separately, and that these letters remain
ordinary text while editing a command. The GPU acceptance run also presses these
keys to suspend and resume a real registered MPS CUDA client.

## Simple sweeps (0.2.2)

Automatic `--name value` arguments, fixed values, repeated cases, optional unique
output folders, and literal arguments containing spaces are checked against real
child scripts. Existing custom argument functions remain covered by the earlier
sweep tests. The manager and real PTY suite has 21 passing tests. Unsupported
automatic value types and conflicting output arguments fail before submission.
