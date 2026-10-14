//! Persistent external-script queues, independent of studies and recordings.
//!
//! The Python client, CLI, and terminal viewer share this one owner process.
//! MPS is an execution adapter; queue control never interprets scientific state.

mod model;
mod mps;
mod server;
mod wire;

pub(crate) use model::process_identity;
pub(crate) use model::{Job, Parent, Status, discover};
pub(crate) use server::tail;
pub(crate) use wire::request;
pub(crate) use wire::terminate_parent;

/// Run the packaged script-manager command line.
///
/// `daemon` is used by the Python facade. `tui` attaches without owning jobs,
/// and `expand` exposes the existing Workflow sweep grammar as JSON.
pub fn run_cli() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let command = arguments.first().map(String::as_str).unwrap_or("help");
    match command {
        "daemon" => server::run(std::path::Path::new(
            arguments
                .get(1)
                .ok_or("daemon requires a launch manifest")?,
        ))?,
        "expand" => {
            use std::io::Read as _;
            let mut input = String::new();
            std::io::stdin()
                .take(8 * 1024 * 1024)
                .read_to_string(&mut input)?;
            let value: serde_json::Value = serde_json::from_str(&input)?;
            println!(
                "{}",
                serde_json::to_string(&scientific_workflow::config::expand_parameters(&value)?)?
            );
        }
        "wake" => {
            let path = std::path::Path::new(
                arguments
                    .get(1)
                    .ok_or("wake requires captured parent metadata")?,
            );
            wire::wake_parent(&model::read_json::<Parent>(path)?)?;
        }
        "tui" => {
            let mut root = std::path::PathBuf::from(".mps-manager");
            let mut pid = None;
            let mut index = 1;
            while index < arguments.len() {
                match arguments[index].as_str() {
                    "--state-dir" => {
                        index += 1;
                        root = arguments
                            .get(index)
                            .ok_or("missing state directory")?
                            .into();
                    }
                    "--pid" => {
                        index += 1;
                        pid = Some(arguments.get(index).ok_or("missing parent PID")?.parse()?);
                    }
                    other => return Err(format!("unknown option {other}").into()),
                }
                index += 1;
            }
            crate::tui::run(&root, pid)?;
        }
        "help" | "--help" | "-h" => {
            println!(
                "MPS Manager\n\n  mps-manager tui [--pid PID] [--state-dir PATH]\n\nUse the mps_runner Python API to create titled queues and submit scripts."
            );
        }
        other => return Err(format!("unknown command {other}").into()),
    }
    Ok(())
}
