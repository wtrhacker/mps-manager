//! Host-local service lifetime and atomic, round-robin admission across parents.

use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::model::{Launch, Parent, Result, process_identity, random_id, read_json, write_json};
use super::mps::{Mps, select_gpu, uid};

#[derive(Clone, Serialize, Deserialize)]
struct Member {
    id: String,
    root: PathBuf,
    pid: u32,
    start: u64,
    #[serde(default)]
    order: u64,
}

impl Member {
    fn alive(&self) -> bool {
        process_identity(self.pid)
            .is_some_and(|(state, start, _)| state != 'Z' && state != 'X' && start == self.start)
    }

    fn parent(&self) -> Option<Parent> {
        let parent = read_json::<Parent>(&self.root.join("state.json")).ok()?;
        (parent.id == self.id && parent.pid == self.pid && parent.process_start == self.start)
            .then_some(parent)
    }
}

#[derive(Serialize, Deserialize)]
struct Slot {
    parent: String,
    root: PathBuf,
    job: String,
    pid: Option<u32>,
    start: Option<u64>,
}

#[derive(Serialize, Deserialize)]
struct Session {
    generation: String,
    launch: Launch,
    external: bool,
    uuid: Option<String>,
    pipe: Option<PathBuf>,
    pid: u32,
    start: u64,
    ready: bool,
    error: Option<String>,
    limit: usize,
    members: Vec<Member>,
    slots: Vec<Slot>,
    last: Option<String>,
    #[serde(default)]
    last_order: Option<u64>,
    #[serde(default)]
    registrations: u64,
    sequence: u64,
    extra_clients: usize,
}

impl Session {
    fn alive(&self) -> bool {
        process_identity(self.pid)
            .is_some_and(|(state, start, _)| state != 'Z' && state != 'X' && start == self.start)
    }

    fn prune(&mut self) {
        self.members.retain(Member::alive);
        // A slot is reusable only after the parent has recorded completed teardown.
        // Keep unknown/orphaned reservations rather than oversubscribe a GPU.
        self.slots.retain(|slot| {
            !read_json::<Parent>(&slot.root.join("state.json"))
                .ok()
                .and_then(|p| p.jobs.into_iter().find(|j| j.id == slot.job))
                .is_some_and(|job| job.state.terminal())
        });
    }

    fn next(&self) -> Option<&Member> {
        let begin = self
            .last_order
            .and_then(|last| self.members.iter().position(|m| m.order > last))
            .unwrap_or(0);
        (0..self.members.len()).find_map(|offset| {
            let member = &self.members[(begin + offset) % self.members.len()];
            let parent = member.parent()?;
            let stopped =
                process_identity(member.pid).is_some_and(|(s, _, _)| s == 'T' || s == 't');
            let reserved = self.slots.iter().filter(|s| s.parent == member.id).count();
            (member.alive()
                && !stopped
                && parent.state == "ready"
                && !parent.admission_paused
                && !parent.pending.is_empty()
                && reserved < parent.max_concurrent)
                .then_some(member)
        })
    }
}

pub(super) struct Coordinator {
    root: PathBuf,
    generation: String,
    parent: String,
    left: bool,
    pub limit: usize,
    pub pid: u32,
    pub uuid: Option<String>,
    pub pipe: Option<PathBuf>,
}

fn lock(root: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join("lock"))?;
    file.lock()?;
    Ok(file)
}

fn session_path(root: &Path) -> PathBuf {
    root.join("session.json")
}

fn digest(value: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn private_directory(root: &Path) -> Result<()> {
    match fs::create_dir(root) {
        Ok(()) => fs::set_permissions(root, fs::Permissions::from_mode(0o700))?,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }
    let metadata = fs::symlink_metadata(root)?;
    if !metadata.is_dir() || metadata.uid() != uid() || metadata.mode() & 0o077 != 0 {
        return Err("shared service directory must be private and owned by this user".into());
    }
    Ok(())
}

impl Coordinator {
    pub fn join(launch: &Launch, parent: &Parent) -> Result<Self> {
        let uuid = if launch.mps {
            Some(select_gpu(&launch.gpu)?)
        } else {
            None
        };
        let explicit = launch.mps_pipe_directory.clone().or_else(|| {
            std::env::var_os("CUDA_MPS_PIPE_DIRECTORY")
                .filter(|p| !p.is_empty())
                .map(PathBuf::from)
        });
        let key = uuid.clone().unwrap_or_else(|| {
            format!(
                "cpu-{:x}-{}",
                digest(&launch.state_dir.to_string_lossy()),
                launch.gpu
            )
        });
        let mut root = std::env::temp_dir().join(format!("mps-manager-v2-{}-{key}", uid()));
        // Explicit attachment to our own managed endpoint joins its existing scheduler.
        if let Some(pipe) = &explicit {
            let managed = read_json::<Session>(&session_path(&root)).ok();
            if !managed.is_some_and(|s| s.alive() && s.pipe.as_ref() == Some(pipe)) {
                root = std::env::temp_dir().join(format!(
                    "mps-manager-v2-{}-{key}-{:x}",
                    uid(),
                    digest(&pipe.to_string_lossy())
                ));
            }
        }
        private_directory(&root)?;
        {
            let _guard = lock(&root)?;
            let existing = read_json::<Session>(&session_path(&root)).ok();
            let mut session = if let Some(mut session) = existing {
                session.prune();
                if session.alive() {
                    if session.error.is_some() {
                        return Err(session.error.unwrap_or_default().into());
                    }
                    session
                } else {
                    if !session.members.is_empty() || !session.slots.is_empty() {
                        return Err("shared service exited with unfinished parents; inspect their jobs before starting another service".into());
                    }
                    Self::initial(launch, uuid.clone(), explicit.clone())?
                }
            } else {
                Self::initial(launch, uuid.clone(), explicit.clone())?
            };
            if launch
                .gpu_max_concurrent
                .is_some_and(|limit| limit != session.limit)
            {
                return Err(format!(
                    "GPU service already uses gpu_max_concurrent={}; omit the option to inherit it",
                    session.limit
                )
                .into());
            }
            session.registrations += 1;
            session.members.push(Member {
                id: parent.id.clone(),
                root: parent.root.clone(),
                pid: parent.pid,
                start: parent.process_start,
                order: session.registrations,
            });
            write_json(&session_path(&root), &session)?;
            if session.pid == 0 {
                let log = File::create(root.join("service.log"))?;
                let mut child = Command::new(std::env::current_exe()?)
                    .arg("service")
                    .arg(&root)
                    .env_remove("CUDA_MPS_PIPE_DIRECTORY")
                    .stdin(Stdio::null())
                    .stdout(Stdio::from(log.try_clone()?))
                    .stderr(Stdio::from(log))
                    .process_group(0)
                    .spawn()?;
                let (_, start, _) =
                    process_identity(child.id()).ok_or("cannot identify shared service")?;
                session.pid = child.id();
                session.start = start;
                write_json(&session_path(&root), &session)?;
                // Rust Child does not kill on drop; the service outlives the first parent.
                let _ = child.try_wait()?;
            }
        }
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let session: Session = read_json(&session_path(&root))?;
            if let Some(error) = session.error {
                return Err(error.into());
            }
            if session.ready {
                return Ok(Self {
                    root,
                    generation: session.generation,
                    parent: parent.id.clone(),
                    left: false,
                    limit: session.limit,
                    pid: session.pid,
                    uuid: session.uuid,
                    pipe: session.pipe,
                });
            }
            if !session.alive() || Instant::now() >= deadline {
                return Err("shared GPU service failed to start; inspect its service.log".into());
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    fn initial(launch: &Launch, uuid: Option<String>, pipe: Option<PathBuf>) -> Result<Session> {
        let mut launch = launch.clone();
        if let Some(uuid) = &uuid {
            launch.gpu = uuid.clone();
        }
        launch.mps_pipe_directory = pipe.clone();
        Ok(Session {
            generation: random_id()?,
            external: pipe.is_some(),
            uuid,
            pipe,
            pid: 0,
            start: 0,
            ready: false,
            error: None,
            limit: launch.gpu_max_concurrent.unwrap_or(64),
            launch,
            members: Vec::new(),
            slots: Vec::new(),
            last: None,
            last_order: None,
            registrations: 0,
            sequence: 0,
            extra_clients: 0,
        })
    }

    fn state(&self) -> Result<Session> {
        let session: Session = read_json(&session_path(&self.root))?;
        if session.generation != self.generation || !session.alive() {
            return Err("shared service identity changed or exited".into());
        }
        if let Some(error) = &session.error {
            return Err(error.clone().into());
        }
        Ok(session)
    }

    pub fn health(&self) -> Result<()> {
        self.state().map(|_| ())
    }

    pub fn acquire(&self, job: &str) -> Result<Option<u64>> {
        let _guard = lock(&self.root)?;
        let mut session = self.state()?;
        session.prune();
        if session.slots.len() + session.extra_clients >= session.limit
            || !session.next().is_some_and(|m| m.id == self.parent)
        {
            return Ok(None);
        }
        let member = session
            .members
            .iter()
            .find(|m| m.id == self.parent)
            .ok_or("parent left shared service")?;
        let parent = member.parent().ok_or("parent metadata is unavailable")?;
        if parent.pending.first().map(String::as_str) != Some(job) {
            return Ok(None);
        }
        session.slots.push(Slot {
            parent: self.parent.clone(),
            root: member.root.clone(),
            job: job.into(),
            pid: None,
            start: None,
        });
        session.last = Some(self.parent.clone());
        session.last_order = Some(member.order);
        session.sequence += 1;
        let sequence = session.sequence;
        write_json(&session_path(&self.root), &session)?;
        Ok(Some(sequence))
    }

    pub fn bind(&self, job: &str, pid: u32, start: u64) -> Result<()> {
        let _guard = lock(&self.root)?;
        let mut session = self.state()?;
        let slot = session
            .slots
            .iter_mut()
            .find(|s| s.parent == self.parent && s.job == job)
            .ok_or("GPU admission reservation was lost")?;
        slot.pid = Some(pid);
        slot.start = Some(start);
        write_json(&session_path(&self.root), &session)
    }

    pub fn leave(&mut self) -> Result<()> {
        if self.left {
            return Ok(());
        }
        let _guard = lock(&self.root)?;
        let mut session: Session = read_json(&session_path(&self.root))?;
        if session.generation != self.generation {
            return Err("shared service identity changed".into());
        }
        session.members.retain(|m| m.id != self.parent);
        session.slots.retain(|s| s.parent != self.parent);
        write_json(&session_path(&self.root), &session)?;
        self.left = true;
        Ok(())
    }
}

/// The service owns MPS, not jobs. Parents retain all per-job controls and metadata.
pub(super) fn run(root: &Path) -> Result<()> {
    unsafe {
        libc::umask(0o077);
    }
    let initial: Session = {
        let _guard = lock(root)?;
        read_json(&session_path(root))?
    };
    let result = (|| -> Result<()> {
        let mut mps = if initial.launch.mps {
            Some(Mps::start(&initial.launch)?)
        } else {
            None
        };
        if !initial.external
            && let Some(mps) = &mut mps
        {
            super::server::probe(&initial.launch, root, mps)?;
        }
        {
            let _guard = lock(root)?;
            let mut session: Session = read_json(&session_path(root))?;
            session.ready = true;
            session.pipe = mps.as_ref().map(|m| m.pipe.clone());
            write_json(&session_path(root), &session)?;
        }
        let mut inspected = Instant::now();
        loop {
            let clients = if inspected.elapsed() >= Duration::from_millis(200) {
                inspected = Instant::now();
                if let Some(mps) = &mut mps {
                    mps.health()?;
                    Some(mps.clients()?)
                } else {
                    None
                }
            } else {
                None
            };
            {
                let _guard = lock(root)?;
                let mut session: Session = read_json(&session_path(root))?;
                session.prune();
                if let Some(clients) = &clients {
                    session.extra_clients = clients
                        .iter()
                        .filter(|(_, pid)| !session.slots.iter().any(|s| s.pid == Some(*pid)))
                        .count();
                }
                if session.members.is_empty() && session.slots.is_empty() {
                    // Serialize final shutdown with registration of a new parent.
                    let no_clients = if session.external {
                        true
                    } else if let Some(mps) = &mps {
                        mps.clients()?.is_empty()
                    } else {
                        true
                    };
                    if no_clients {
                        if let Some(mps) = &mut mps {
                            mps.stop()?;
                        }
                        drop(mps.take());
                        session.ready = false;
                        // Mark stopped while holding the registration lock, before exiting.
                        session.pid = 0;
                        write_json(&session_path(root), &session)?;
                        return Ok(());
                    }
                }
                write_json(&session_path(root), &session)?;
            }
            thread::sleep(Duration::from_millis(30));
        }
    })();
    if let Err(error) = &result {
        let _guard = lock(root)?;
        let mut session: Session = read_json(&session_path(root))?;
        session.error = Some(error.to_string());
        write_json(&session_path(root), &session)?;
    }
    result
}
