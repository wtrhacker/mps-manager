use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::PermissionsExt as _;
use std::os::unix::net::UnixListener;
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::model::{
    Job, Launch, Parent, Result, Status, Submission, hostname, now, process_identity, read_json,
    write_json,
};
use super::mps::{Mps, socket_directory};
use super::shared::Coordinator;
use super::wire::{receive, same_user, send};
use scientific_workflow::external::{ProcessTree, spawn};

static SHUTDOWN: AtomicBool = AtomicBool::new(false);
extern "C" fn stop_signal(_: libc::c_int) {
    SHUTDOWN.store(true, Ordering::Relaxed);
}

struct Active {
    process: ProcessTree,
    directory: PathBuf,
    admitted: Instant,
    connected: bool,
}

struct Server {
    parent: Parent,
    worker: PathBuf,
    mps: Option<Mps>,
    coordinator: Option<Coordinator>,
    active: BTreeMap<String, Active>,
    shutting_down: bool,
    dismiss_after_shutdown: bool,
}

pub(super) fn run(manifest: &Path) -> Result<()> {
    // This daemon owns its private metadata, sockets, logs, and control files.
    unsafe {
        libc::umask(0o077);
    }
    let launch: Launch = read_json(manifest)?;
    let root = launch.state_dir.join("queues").join(&launch.id);
    fs::create_dir_all(&root)?;
    let outcome = run_inner(launch, &root);
    if let Err(error) = &outcome {
        let _ = write_json(
            &root.join("startup-error.json"),
            &json!({"error": error.to_string()}),
        );
        if let Ok(mut parent) = read_json::<Parent>(&root.join("state.json")) {
            parent.state = "failed".into();
            parent.error = Some(error.to_string());
            let _ = write_json(&root.join("state.json"), &parent);
        }
    }
    outcome
}

fn run_inner(launch: Launch, root: &Path) -> Result<()> {
    if launch.max_concurrent == 0 || launch.max_concurrent > 64 {
        return Err("max_concurrent must be between 1 and 64".into());
    }
    if launch
        .gpu_max_concurrent
        .is_some_and(|limit| limit == 0 || limit > 64)
    {
        return Err("gpu_max_concurrent must be between 1 and 64".into());
    }
    if !launch.worker.is_file() || !launch.python.is_file() {
        return Err("Python interpreter or worker is missing".into());
    }
    let socket = socket_directory()?.join("control.sock");
    let mut parent = Parent {
        schema_version: 1,
        id: launch.id.clone(),
        title: launch.title.clone(),
        pid: std::process::id(),
        process_start: process_identity(std::process::id())
            .ok_or("cannot identify manager process")?
            .1,
        host: hostname(),
        root: root.to_owned(),
        socket: socket.clone(),
        gpu: launch.gpu.clone(),
        gpu_uuid: None,
        mps_pipe_directory: None,
        mps: launch.mps,
        max_concurrent: launch.max_concurrent,
        common_args: launch.common_args.clone(),
        gpu_max_concurrent: None,
        service_pid: None,
        admission_paused: false,
        state: "starting".into(),
        error: None,
        submitted_at: now(),
        jobs: Vec::new(),
        pending: Vec::new(),
        messages: Vec::new(),
        dismissed: false,
    };
    write_json(&root.join("state.json"), &parent)?;
    let coordinator = if launch.mps || launch.gpu_max_concurrent.is_some() {
        Some(Coordinator::join(&launch, &parent)?)
    } else {
        None
    };
    let mps = if launch.mps {
        let service = coordinator.as_ref().ok_or("GPU service is missing")?;
        Some(Mps::connection(
            service.uuid.clone().ok_or("GPU UUID is missing")?,
            service.pipe.clone().ok_or("MPS endpoint is missing")?,
        ))
    } else {
        None
    };
    if let Some(service) = &coordinator {
        parent.gpu_uuid = service.uuid.clone();
        parent.mps_pipe_directory = service.pipe.clone();
        parent.gpu_max_concurrent = Some(service.limit);
        parent.service_pid = Some(service.pid);
    }
    let listener = UnixListener::bind(&socket)?;
    fs::set_permissions(&socket, fs::Permissions::from_mode(0o600))?;
    listener.set_nonblocking(true)?;
    // SAFETY: handlers only set a lock-free flag; cleanup runs on the owner loop.
    unsafe {
        libc::signal(libc::SIGINT, stop_signal as *const () as libc::sighandler_t);
        libc::signal(
            libc::SIGTERM,
            stop_signal as *const () as libc::sighandler_t,
        );
    }
    parent.state = "ready".into();
    parent.message(format!(
        "Queue '{}' is ready; no automatic retries.",
        parent.title
    ));
    let mut server = Server {
        parent,
        worker: launch.worker,
        mps,
        coordinator,
        active: BTreeMap::new(),
        shutting_down: false,
        dismiss_after_shutdown: false,
    };
    server.save()?;
    write_json(
        &root.join("ready.json"),
        &json!({"pid": std::process::id()}),
    )?;
    let mut health = Instant::now();
    loop {
        if SHUTDOWN.load(Ordering::Relaxed) {
            server.shutting_down = true;
        }
        for _ in 0..32 {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
                    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
                    let result = (|| {
                        if !same_user(&stream) {
                            return Err("manager belongs to a different user".into());
                        }
                        let request = receive(&stream)?;
                        if request["queue_id"] != server.parent.id {
                            return Err("queue identity changed; reconnect".into());
                        }
                        server.handle(request)
                    })();
                    let reply = match result {
                        Ok(result) => json!({"ok": true, "result": result}),
                        Err(error) => json!({"ok": false, "error": error.to_string()}),
                    };
                    let _ = send(&mut stream, &reply);
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error.into()),
            }
        }
        if let Err(error) = server.tick() {
            server.parent.error = Some(error.to_string());
            server.parent.state = "fault".into();
            server.parent.admission_paused = true;
            server.parent.message(format!("MPS/process fault: {error}"));
            server.save()?;
        }
        if health.elapsed() >= Duration::from_secs(1) {
            if let Some(service) = &server.coordinator
                && let Err(error) = service.health()
            {
                server.parent.error = Some(error.to_string());
                server.parent.state = "fault".into();
                server.parent.admission_paused = true;
            }
            if let Some(mps) = &mut server.mps {
                if let Err(error) = mps.health() {
                    server.parent.error = Some(error.to_string());
                    server.parent.state = "fault".into();
                    server.parent.admission_paused = true;
                }
                if server.parent.state != "fault" {
                    match mps.clients() {
                        Ok(clients) => {
                            let lost = server.active.values().any(|active| {
                                active.connected
                                    && !active.directory.join("done.json").exists()
                                    && !clients.iter().any(|(_, pid)| *pid == active.process.id())
                            });
                            if lost {
                                server.parent.error =
                                    Some("an active child lost its MPS connection".into());
                                server.parent.state = "fault".into();
                                server.parent.admission_paused = true;
                            }
                        }
                        Err(error) => {
                            server.parent.error = Some(error.to_string());
                            server.parent.state = "fault".into();
                            server.parent.admission_paused = true;
                        }
                    }
                }
            }
            server.save()?;
            health = Instant::now();
        }
        if server.shutting_down {
            server.parent.admission_paused = true;
            let ids: Vec<String> = server.active.keys().cloned().collect();
            let mut complete = true;
            for id in ids {
                if let Err(error) = server.cancel(&id) {
                    complete = false;
                    server.parent.error = Some(error.to_string());
                    server.parent.state = "fault".into();
                }
            }
            for job in &mut server.parent.jobs {
                if job.state == Status::Pending {
                    job.state = Status::Cancelled;
                    job.finished_at = Some(now());
                }
            }
            server.parent.pending.clear();
            if complete && server.active.is_empty() {
                if let Some(service) = &mut server.coordinator {
                    service.leave()?;
                }
                if let Some(mps) = &mut server.mps
                    && let Err(error) = mps.stop()
                {
                    server.parent.error = Some(error.to_string());
                    server.parent.state = "fault".into();
                    server
                        .parent
                        .message(format!("MPS cleanup failed: {error}"));
                    server.shutting_down = false;
                    server.save()?;
                    continue;
                }
                server.parent.state = "closed".into();
                server.parent.dismissed = server.dismiss_after_shutdown;
                server.save()?;
                break;
            }
            server.save()?;
        }
        thread::sleep(Duration::from_millis(30));
    }
    Ok(())
}

pub(super) fn probe(launch: &Launch, root: &Path, mps: &mut Mps) -> Result<()> {
    let directory = root.join("probe");
    fs::create_dir_all(&directory)?;
    let log = File::create(directory.join("probe.log"))?;
    let manifest = directory.join("request.json");
    write_json(
        &manifest,
        &json!({"probe": true, "gpu_uuid": mps.uuid, "directory": directory}),
    )?;
    let mut child = Command::new(&launch.python)
        .args([
            "-u",
            launch.worker.to_str().ok_or("worker path is not UTF-8")?,
            manifest.to_str().ok_or("manifest path is not UTF-8")?,
        ])
        .envs(mps.environment())
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .spawn()?;
    let deadline = Instant::now() + Duration::from_secs(15);
    let outcome = (|| {
        loop {
            if directory.join("ready.json").exists() && mps.attached(child.id())? {
                fs::write(directory.join("go"), b"probe")?;
                return Ok(());
            }
            if child.try_wait()?.is_some() {
                return Err("CUDA/MPS probe failed; inspect probe/probe.log".into());
            }
            if Instant::now() >= deadline {
                return Err("CUDA client did not attach to MPS; refusing fallback".into());
            }
            thread::sleep(Duration::from_millis(30));
        }
    })();
    if outcome.is_err() {
        let _ = fs::write(directory.join("abort"), b"abort");
    }
    let exit_deadline = Instant::now() + Duration::from_secs(5);
    while child.try_wait()?.is_none() {
        if Instant::now() >= exit_deadline {
            if mps.attached(child.id())? {
                let server = mps
                    .clients()?
                    .into_iter()
                    .find(|(_, pid)| *pid == child.id())
                    .ok_or("probe client identity changed")?
                    .0;
                if mps.command(&format!("terminate_client {server} {}", child.id()))? != "0" {
                    return Err("probe GPU teardown failed".into());
                }
            }
            child.kill()?;
            child.wait()?;
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    outcome
}

impl Server {
    fn save(&self) -> Result<()> {
        write_json(&self.parent.root.join("state.json"), &self.parent)
    }

    fn job_mut(&mut self, id: &str) -> Result<&mut Job> {
        self.parent
            .jobs
            .iter_mut()
            .find(|job| job.id == id)
            .ok_or_else(|| "unknown job ID".into())
    }

    fn handle(&mut self, request: Value) -> Result<Value> {
        let action = request["action"].as_str().ok_or("missing action")?;
        let id = request["job_id"].as_str().unwrap_or("");
        match action {
            "status" => return Ok(serde_json::to_value(&self.parent)?),
            "job" => return Ok(serde_json::to_value(self.job_mut(id)?)?),
            "common_args" => {
                self.parent.common_args = serde_json::from_value(request["args"].clone())?;
            }
            "submit" => {
                if self.shutting_down || self.parent.state == "fault" {
                    return Err("queue is closing or faulted; no jobs admitted".into());
                }
                let raw = request["runs"].as_array().ok_or("runs must be an array")?;
                if raw.is_empty() {
                    return Ok(json!([]));
                }
                let mut jobs = Vec::new();
                for value in raw {
                    let spec: Submission = serde_json::from_value(value.clone())?;
                    if !spec.id.starts_with("job-")
                        || !spec.id[4..].chars().all(|c| c.is_ascii_hexdigit())
                        || spec.id.len() < 12
                    {
                        return Err("invalid job ID".into());
                    }
                    if self.parent.jobs.iter().any(|j| j.id == spec.id)
                        || jobs.iter().any(|j: &Job| j.id == spec.id)
                    {
                        return Err(
                            "duplicate job ID; reconnect and inspect the existing submission"
                                .into(),
                        );
                    }
                    if !spec.script_path.is_absolute()
                        || !spec.script_path.is_file()
                        || !spec.cwd.is_dir()
                        || !spec.python.is_file()
                    {
                        return Err(
                            "script, interpreter, or working directory is unavailable".into()
                        );
                    }
                    if spec.env.keys().any(|key| {
                        matches!(
                            key.as_str(),
                            "CUDA_VISIBLE_DEVICES"
                                | "CUDA_MPS_PIPE_DIRECTORY"
                                | "CUDA_MPS_LOG_DIRECTORY"
                                | "CUDA_MPS_PROTOCOL_VERSION"
                                | "MPS_RUN_DIR"
                                | "MPS_JOB_ID"
                        ) || key.starts_with("MPS_MANAGER_")
                    }) {
                        return Err(
                            "GPU and manager environment variables are owned by the queue".into(),
                        );
                    }
                    let title = spec.title.unwrap_or_else(|| {
                        spec.script_path
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned()
                    });
                    validate_title(&title)?;
                    let run_dir = self
                        .parent
                        .root
                        .join("jobs")
                        .join(&spec.id)
                        .join("artifacts");
                    fs::create_dir_all(&run_dir)?;
                    write_json(
                        &run_dir
                            .parent()
                            .ok_or("missing job directory")?
                            .join("environment.json"),
                        &spec.env,
                    )?;
                    let common_args = spec
                        .common_args
                        .unwrap_or_else(|| self.parent.common_args.clone());
                    let mut args = common_args.clone();
                    args.extend(spec.args.clone());
                    jobs.push(Job {
                        id: spec.id,
                        title,
                        script_path: spec.script_path,
                        args,
                        common_args,
                        submitted_args: spec.args,
                        parameters: spec.parameters,
                        admission_sequence: None,
                        python: spec.python,
                        cwd: spec.cwd,
                        env: spec.env,
                        run_dir,
                        state: Status::Pending,
                        pid: None,
                        process_start: None,
                        exit_code: None,
                        submitted_at: now(),
                        started_at: None,
                        finished_at: None,
                        detail: String::new(),
                    });
                }
                let ids: Vec<String> = jobs.iter().map(|j| j.id.clone()).collect();
                self.parent.pending.extend(ids.clone());
                self.parent.jobs.extend(jobs);
                self.parent.message(format!("Added {} job(s).", ids.len()));
                self.save()?;
                return Ok(json!(ids));
            }
            "pause" | "resume" => {
                let paused = action == "pause";
                if id.is_empty() {
                    self.parent.admission_paused = paused;
                    let ids: Vec<String> = self.active.keys().cloned().collect();
                    for id in ids {
                        self.suspend(&id, paused)?;
                    }
                } else {
                    let state = self.job_mut(id)?.state;
                    if state == Status::Pending {
                        if paused {
                            self.parent.pending.retain(|value| value != id);
                            self.job_mut(id)?.detail = "held".into();
                        } else {
                            if !self.parent.pending.iter().any(|value| value == id) {
                                self.parent.pending.push(id.into());
                            }
                            self.job_mut(id)?.detail.clear();
                        }
                    } else {
                        self.suspend(id, paused)?;
                    }
                }
            }
            "hold" | "release" => {
                if action == "hold" {
                    if self.job_mut(id)?.state != Status::Pending {
                        return Err("only pending jobs can be held".into());
                    }
                    self.parent.pending.retain(|value| value != id);
                    self.job_mut(id)?.detail = "held".into();
                } else {
                    if self.job_mut(id)?.state != Status::Pending {
                        return Err("only pending jobs can be released".into());
                    }
                    if !self.parent.pending.iter().any(|value| value == id) {
                        self.parent.pending.push(id.into());
                    }
                    self.job_mut(id)?.detail.clear();
                }
            }
            "kill" => {
                if id.is_empty() {
                    return Err("kill requires a job ID".into());
                }
                self.cancel(id)?;
            }
            "remove" => {
                let prior = self.job_mut(id)?.state;
                if prior != Status::Pending && !prior.terminal() {
                    return Err("cancel a running job before removing it".into());
                }
                self.parent.pending.retain(|value| value != id);
                let job = self.job_mut(id)?;
                job.state = Status::Removed;
                job.finished_at = Some(now());
                job.detail = format!("removed from queue; prior state: {}", prior.label());
            }
            "move" => {
                let position = request["position"]
                    .as_u64()
                    .ok_or("position must be a nonnegative integer")?
                    as usize;
                if !self.parent.pending.iter().any(|value| value == id) {
                    return Err("only runnable pending jobs can be reordered".into());
                }
                self.parent.pending.retain(|value| value != id);
                self.parent
                    .pending
                    .insert(position.min(self.parent.pending.len()), id.to_owned());
            }
            "reorder" => {
                let ids: Vec<String> = serde_json::from_value(request["job_ids"].clone())?;
                let mut expected = self.parent.pending.clone();
                expected.sort();
                let mut actual = ids.clone();
                actual.sort();
                if expected != actual {
                    return Err(
                        "reorder must include each runnable pending job exactly once".into(),
                    );
                }
                self.parent.pending = ids;
            }
            "shuffle" => {
                let mut random = File::open("/dev/urandom")?;
                for index in (1..self.parent.pending.len()).rev() {
                    let mut bytes = [0u8; 8];
                    random.read_exact(&mut bytes)?;
                    let other = (u64::from_ne_bytes(bytes) % (index as u64 + 1)) as usize;
                    self.parent.pending.swap(index, other);
                }
            }
            "title" => {
                let title = request["title"].as_str().ok_or("missing title")?;
                validate_title(title)?;
                if id.is_empty() {
                    self.parent.title = title.into();
                } else {
                    self.job_mut(id)?.title = title.into();
                }
            }
            "logs" => {
                let job = self.job_mut(id)?;
                let name = if request["stream"] == "stdout" {
                    "stdout.log"
                } else {
                    "stderr.log"
                };
                return Ok(json!(tail(
                    &job.run_dir
                        .parent()
                        .ok_or("missing job directory")?
                        .join(name)
                )?));
            }
            "shutdown" => {
                self.shutting_down = true;
                self.parent.state = "closing".into();
                self.parent.error = None;
            }
            "terminate" => {
                self.shutting_down = true;
                self.dismiss_after_shutdown = true;
                self.parent.state = "closing".into();
                self.parent.error = None;
            }
            _ => return Err(format!("unknown manager action {action}").into()),
        }
        self.parent.message(format!(
            "{action} {}",
            if id.is_empty() { "queue" } else { id }
        ));
        self.save()?;
        Ok(serde_json::to_value(&self.parent)?)
    }

    fn suspend(&mut self, id: &str, paused: bool) -> Result<()> {
        let active = self.active.get_mut(id).ok_or("job is not running")?;
        if !active.connected {
            self.job_mut(id)?.detail = if paused {
                "pause requested".into()
            } else {
                String::new()
            };
            return Ok(());
        }
        active.process.force_suspension(paused)?;
        self.job_mut(id)?.state = if paused {
            Status::Suspended
        } else {
            Status::Running
        };
        Ok(())
    }

    fn cancel(&mut self, id: &str) -> Result<()> {
        if self.job_mut(id)?.state.terminal() {
            return Ok(());
        }
        // NVIDIA's termination handshake needs the client's driver threads to
        // run. Resuming it is safe: terminate_client handles in-flight GPU work.
        if let Some(active) = self.active.get_mut(id) {
            active.process.force_suspension(false)?;
        }
        if let Some(active) = self.active.get(id)
            && let Some(mps) = &self.mps
        {
            mps.terminate_group(active.process.id())?;
        }
        if let Some(mut active) = self.active.remove(id) {
            active.process.force_suspension(false)?;
            active.process.stop_after_teardown();
        }
        self.parent.pending.retain(|value| value != id);
        let job = self.job_mut(id)?;
        job.state = Status::Cancelled;
        job.finished_at = Some(now());
        job.detail = "cancelled explicitly; never retried automatically".into();
        self.save()?;
        Ok(())
    }

    fn tick(&mut self) -> Result<()> {
        let ids: Vec<String> = self.active.keys().cloned().collect();
        let mut changed = false;
        for id in ids {
            let active = self.active.get_mut(&id).ok_or("active job disappeared")?;
            if let Some(status) = active.process.try_wait()? {
                let done = read_json::<Value>(&active.directory.join("done.json")).ok();
                let code = done
                    .as_ref()
                    .and_then(|value| value["exit_code"].as_i64())
                    .map(|code| code as i32)
                    .or(status.code());
                let mut active = self.active.remove(&id).ok_or("active job disappeared")?;
                if let Some(mps) = &self.mps {
                    mps.terminate_group(active.process.id())?;
                }
                active.process.stop_after_teardown();
                let job = self.job_mut(&id)?;
                job.state = if code == Some(0) {
                    Status::Succeeded
                } else {
                    Status::Failed
                };
                job.exit_code = code;
                job.finished_at = Some(now());
                if !active.connected && self.mps.is_some() {
                    self.parent.error = Some(format!(
                        "{id} failed CUDA/MPS initialization; scientific code was not started"
                    ));
                    self.parent.admission_paused = true;
                    self.parent.state = "fault".into();
                } else if done.is_none() && self.mps.is_some() {
                    self.parent.message(format!("{id} exited outside the manager handshake; GPU service health will be checked."));
                }
                changed = true;
                continue;
            }
            if !active.connected {
                if active.directory.join("ready.json").exists() {
                    let attached = if let Some(mps) = &self.mps {
                        mps.attached(active.process.id())?
                    } else {
                        true
                    };
                    if attached {
                        fs::write(active.directory.join("go"), b"run")?;
                        active.connected = true;
                        let paused = self.parent.admission_paused
                            || self
                                .parent
                                .jobs
                                .iter()
                                .any(|job| job.id == id && job.detail == "pause requested");
                        if paused {
                            active.process.force_suspension(true)?;
                        }
                        self.job_mut(&id)?.state = if paused {
                            Status::Suspended
                        } else {
                            Status::Running
                        };
                        changed = true;
                    }
                }
                if !self.active.get(&id).is_some_and(|active| active.connected)
                    && self.active[&id].admitted.elapsed() > Duration::from_secs(15)
                {
                    fs::write(self.active[&id].directory.join("abort"), b"abort")?;
                    return Err(
                        "child did not attach to MPS; scientific code was not started".into(),
                    );
                }
            } else if active.directory.join("done.json").exists() {
                let done: Value = read_json(&active.directory.join("done.json"))?;
                if let Some(mps) = &self.mps {
                    mps.terminate_group(active.process.id())?;
                }
                let mut active = self.active.remove(&id).ok_or("active job disappeared")?;
                fs::write(active.directory.join("exit"), b"exit")?;
                active.process.stop_after_teardown();
                let job = self.job_mut(&id)?;
                job.exit_code = done["exit_code"].as_i64().map(|code| code as i32);
                job.state = if job.exit_code == Some(0) {
                    Status::Succeeded
                } else {
                    Status::Failed
                };
                job.finished_at = Some(now());
                job.detail = done["error"].as_str().unwrap_or("").into();
                let state = self.job_mut(&id)?.state.label();
                self.parent.message(format!("{id}: {state}"));
                changed = true;
            } else if let Some((state, start, _)) = process_identity(active.process.id()) {
                let job = self.job_mut(&id)?;
                if job.process_start != Some(start) {
                    return Err("child process identity changed".into());
                }
                let observed = if state == 'T' || state == 't' {
                    Status::Suspended
                } else {
                    Status::Running
                };
                if job.state != observed {
                    job.state = observed;
                    changed = true;
                }
            }
        }
        if changed {
            self.save()?;
        }
        while !self.shutting_down
            && !self.parent.admission_paused
            && self.active.len() < self.parent.max_concurrent
            && !self.parent.pending.is_empty()
        {
            let id = self.parent.pending[0].clone();
            if let Some(service) = &self.coordinator {
                let Some(sequence) = service.acquire(&id)? else {
                    break;
                };
                self.job_mut(&id)?.admission_sequence = Some(sequence);
            }
            self.parent.pending.remove(0);
            if let Err(error) = self.launch(&id) {
                let job = self.job_mut(&id)?;
                job.state = Status::Failed;
                job.finished_at = Some(now());
                job.detail = error.to_string();
                self.parent
                    .message(format!("{id} failed to launch: {error}; no retry."));
            }
            changed = true;
        }
        if changed {
            self.save()?;
        }
        Ok(())
    }

    fn launch(&mut self, id: &str) -> Result<()> {
        let mut job = self.job_mut(id)?.clone();
        let directory = job
            .run_dir
            .parent()
            .ok_or("missing job directory")?
            .to_owned();
        fs::create_dir_all(&directory)?;
        let manifest = directory.join("request.json");
        write_json(
            &manifest,
            &json!({"script_path": job.script_path, "args": job.args,
            "directory": directory, "gpu_uuid": self.parent.gpu_uuid, "probe": false}),
        )?;
        job.state = Status::Starting;
        job.started_at = Some(now());
        *self.job_mut(id)? = job.clone();
        self.save()?;
        let stdout = File::create(directory.join("stdout.log"))?;
        let stderr = File::create(directory.join("stderr.log"))?;
        let mut command = Command::new(&job.python);
        command
            .arg("-u")
            .arg(&self.worker)
            .arg(&manifest)
            .current_dir(&job.cwd)
            .envs(&job.env)
            .env("MPS_RUN_DIR", &job.run_dir)
            .env("MPS_JOB_ID", id)
            .env("MPS_MANAGER_PID", self.parent.pid.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .process_group(0);
        if let Some(mps) = &self.mps {
            command.envs(mps.environment());
        }
        let mut process = spawn(&mut command)?;
        let process_pid = process.id();
        let identity = process_identity(process.id()).ok_or("cannot identify launched child")?;
        job.pid = Some(process.id());
        job.process_start = Some(identity.1);
        if self.mps.is_some() {
            process.disarm_cleanup();
        }
        self.active.insert(
            id.into(),
            Active {
                process,
                directory,
                admitted: Instant::now(),
                connected: false,
            },
        );
        *self.job_mut(id)? = job;
        self.save()?;
        if let Some(service) = &self.coordinator {
            service.bind(id, process_pid, identity.1)?;
        }
        Ok(())
    }
}

fn validate_title(title: &str) -> Result<()> {
    if title.trim().is_empty() || title.len() > 512 || title.chars().any(char::is_control) {
        return Err("title must be nonempty text without control characters".into());
    }
    Ok(())
}

pub(crate) fn tail(path: &Path) -> Result<String> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(String::new()),
        Err(error) => return Err(error.into()),
    };
    let length = file.metadata()?.len();
    file.seek(SeekFrom::Start(length.saturating_sub(65536)))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().rev().take(200).collect();
    Ok(lines.into_iter().rev().collect::<Vec<_>>().join("\n"))
}
