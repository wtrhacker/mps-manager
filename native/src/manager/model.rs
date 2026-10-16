use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

pub(super) type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Status {
    Pending,
    Starting,
    Running,
    Suspended,
    Succeeded,
    Failed,
    Cancelled,
    Removed,
}

impl Status {
    pub(crate) fn terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Cancelled | Self::Removed
        )
    }
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Starting => "starting",
            Self::Running => "running",
            Self::Suspended => "suspended",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Removed => "removed",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Job {
    pub(crate) id: String,
    pub(crate) title: String,
    pub(crate) script_path: PathBuf,
    pub(crate) args: Vec<String>,
    #[serde(default)]
    pub(crate) common_args: Vec<String>,
    #[serde(default)]
    pub(crate) submitted_args: Vec<String>,
    #[serde(default)]
    pub(crate) parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) admission_sequence: Option<u64>,
    #[serde(default)]
    pub(crate) admission_clients: Option<usize>,
    #[serde(default)]
    pub(crate) active_thread_percentage: Option<f64>,
    pub(crate) python: PathBuf,
    pub(crate) cwd: PathBuf,
    #[serde(default, skip_serializing)]
    pub(crate) env: BTreeMap<String, String>,
    pub(crate) run_dir: PathBuf,
    pub(crate) state: Status,
    pub(crate) pid: Option<u32>,
    pub(crate) process_start: Option<u64>,
    pub(crate) exit_code: Option<i32>,
    pub(crate) submitted_at: u64,
    pub(crate) started_at: Option<u64>,
    pub(crate) finished_at: Option<u64>,
    pub(crate) detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Parent {
    pub(crate) schema_version: u32,
    pub(crate) id: String,
    pub(crate) title: String,
    pub(crate) pid: u32,
    pub(crate) process_start: u64,
    pub(crate) host: String,
    pub(crate) root: PathBuf,
    pub(crate) socket: PathBuf,
    pub(crate) gpu: String,
    pub(crate) gpu_uuid: Option<String>,
    pub(crate) mps_pipe_directory: Option<PathBuf>,
    pub(crate) mps: bool,
    pub(crate) max_concurrent: usize,
    #[serde(default)]
    pub(crate) common_args: Vec<String>,
    #[serde(default)]
    pub(crate) gpu_max_concurrent: Option<usize>,
    #[serde(default)]
    pub(crate) service_pid: Option<u32>,
    pub(crate) admission_paused: bool,
    pub(crate) state: String,
    pub(crate) error: Option<String>,
    pub(crate) submitted_at: u64,
    pub(crate) jobs: Vec<Job>,
    pub(crate) pending: Vec<String>,
    pub(crate) messages: Vec<String>,
    #[serde(default)]
    pub(crate) dismissed: bool,
}

impl Parent {
    pub(crate) fn alive(&self) -> bool {
        self.host == hostname()
            && process_identity(self.pid).is_some_and(|(state, start, _)| {
                state != 'Z' && state != 'X' && start == self.process_start
            })
    }
    pub(crate) fn counts(&self) -> (usize, usize, usize, usize) {
        (
            self.jobs
                .iter()
                .filter(|j| j.state == Status::Pending)
                .count(),
            self.jobs
                .iter()
                .filter(|j| {
                    matches!(
                        j.state,
                        Status::Starting | Status::Running | Status::Suspended
                    )
                })
                .count(),
            self.jobs
                .iter()
                .filter(|j| j.state == Status::Succeeded)
                .count(),
            self.jobs
                .iter()
                .filter(|j| j.state == Status::Failed)
                .count(),
        )
    }
    pub(super) fn message(&mut self, message: impl Into<String>) {
        self.messages.push(message.into());
        if self.messages.len() > 100 {
            self.messages.remove(0);
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Launch {
    pub id: String,
    pub title: String,
    pub state_dir: PathBuf,
    pub worker: PathBuf,
    pub python: PathBuf,
    pub gpu: String,
    pub max_concurrent: usize,
    #[serde(default)]
    pub common_args: Vec<String>,
    #[serde(default)]
    pub gpu_max_concurrent: Option<usize>,
    pub mps: bool,
    #[serde(default)]
    pub mps_pipe_directory: Option<PathBuf>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Submission {
    pub id: String,
    pub script_path: PathBuf,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub common_args: Option<Vec<String>>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
    pub python: PathBuf,
    pub cwd: PathBuf,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub title: Option<String>,
}

pub(super) fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub(crate) fn hostname() -> String {
    let mut buffer = [0u8; 256];
    // SAFETY: the buffer is writable and its length is passed to gethostname.
    unsafe {
        libc::gethostname(buffer.as_mut_ptr().cast(), buffer.len());
    }
    let end = buffer
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(buffer.len());
    String::from_utf8_lossy(&buffer[..end]).into_owned()
}

pub(crate) fn process_identity(pid: u32) -> Option<(char, u64, u32)> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let fields: Vec<&str> = stat[stat.rfind(')')? + 2..].split_whitespace().collect();
    Some((
        fields.first()?.chars().next()?,
        fields.get(19)?.parse().ok()?,
        fields.get(2)?.parse().ok()?,
    ))
}

pub(crate) fn read_json<T: for<'a> Deserialize<'a>>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

pub(super) fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let temporary = path.with_extension(format!("{}.partial", random_id()?));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub(super) fn random_id() -> Result<String> {
    let mut bytes = [0u8; 12];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

pub(crate) fn discover(root: &Path) -> Vec<Parent> {
    let mut parents = Vec::new();
    if let Ok(entries) = fs::read_dir(root.join("queues")) {
        for entry in entries.flatten() {
            if let Ok(parent) = read_json::<Parent>(&entry.path().join("state.json"))
                && !parent.dismissed
            {
                parents.push(parent);
            }
        }
    }
    parents.sort_by_key(|p| {
        (
            std::cmp::Reverse(p.alive()),
            std::cmp::Reverse(p.submitted_at),
        )
    });
    parents
}
