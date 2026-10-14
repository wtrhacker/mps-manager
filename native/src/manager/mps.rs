//! CUDA/MPS lifecycle adapter. Ordinary Workflow process cleanup stays generic.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use super::model::{Launch, Result, process_identity, random_id};

pub(super) struct Mps {
    pub uuid: String,
    pub pipe: PathBuf,
    env: BTreeMap<String, String>,
    daemon: Option<Child>,
    _lease: Option<File>,
}

impl Mps {
    pub fn start(launch: &Launch) -> Result<Self> {
        let uuid = select_gpu(&launch.gpu)?;
        let pipe = launch.mps_pipe_directory.clone().or_else(|| {
            std::env::var_os("CUDA_MPS_PIPE_DIRECTORY")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        });
        let owns = pipe.is_none();
        let pipe = pipe.unwrap_or_else(|| {
            std::env::temp_dir().join(format!("wf-mps-{}-{}", std::process::id(), launch.id))
        });
        fs::create_dir_all(&pipe)?;
        let logs = pipe.join("logs");
        fs::create_dir_all(&logs)?;
        let env = BTreeMap::from([
            ("CUDA_VISIBLE_DEVICES".into(), uuid.clone()),
            (
                "CUDA_MPS_PIPE_DIRECTORY".into(),
                pipe.to_string_lossy().into_owned(),
            ),
            (
                "CUDA_MPS_LOG_DIRECTORY".into(),
                logs.to_string_lossy().into_owned(),
            ),
            ("CUDA_MPS_PROTOCOL_VERSION".into(), "2".into()),
        ]);
        let mut mps = Self {
            uuid,
            pipe,
            env,
            daemon: None,
            _lease: None,
        };
        if owns {
            let lock_path =
                std::env::temp_dir().join(format!("wf-mps-{}-{}.lock", uid(), mps.uuid));
            let lock = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(lock_path)?;
            lock.try_lock().map_err(|_| "another manager owns this GPU; connect to its queue or explicitly attach its MPS endpoint")?;
            mps._lease = Some(lock);
            let log = File::create(
                launch
                    .state_dir
                    .join("queues")
                    .join(&launch.id)
                    .join("mps-service.log"),
            )?;
            mps.daemon = Some(
                Command::new("nvidia-cuda-mps-control")
                    .arg("-f")
                    .envs(&mps.env)
                    .stdin(Stdio::null())
                    .stdout(Stdio::from(log.try_clone()?))
                    .stderr(Stdio::from(log))
                    .spawn()
                    .map_err(|error| format!("cannot start NVIDIA MPS: {error}"))?,
            );
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if mps.command("get_server_list").is_ok() {
                break;
            }
            if mps
                .daemon
                .as_mut()
                .is_some_and(|child| child.try_wait().ok().flatten().is_some())
            {
                return Err(
                    "NVIDIA MPS service exited during startup; inspect mps-service.log".into(),
                );
            }
            if Instant::now() >= deadline {
                return Err("cannot connect to the NVIDIA MPS service".into());
            }
            thread::sleep(Duration::from_millis(50));
        }
        Ok(mps)
    }

    pub fn environment(&self) -> &BTreeMap<String, String> {
        &self.env
    }

    pub fn command(&self, input: &str) -> Result<String> {
        let mut child = Command::new("nvidia-cuda-mps-control")
            .envs(&self.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        if let Some(mut stdin) = child.stdin.take() {
            writeln!(stdin, "{input}")?;
        }
        let output = wait_output(child, Duration::from_secs(5))?;
        if !output.status.success() {
            return Err(format!(
                "MPS control failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )
            .into());
        }
        Ok(String::from_utf8(output.stdout)?.trim().to_owned())
    }

    pub fn clients(&self) -> Result<Vec<(u32, u32)>> {
        let mut clients = Vec::new();
        for server in numbers(&self.command("get_server_list")?) {
            for client in numbers(&self.command(&format!("get_client_list {server}"))?) {
                clients.push((server, client));
            }
        }
        Ok(clients)
    }

    pub fn attached(&self, pid: u32) -> Result<bool> {
        Ok(self.clients()?.iter().any(|(_, client)| *client == pid))
    }

    pub fn terminate_group(&self, group: u32) -> Result<()> {
        for (server, client) in self.clients()? {
            if process_identity(client).is_some_and(|(_, _, owned_group)| owned_group == group) {
                let result = self.command(&format!("terminate_client {server} {client}"))?;
                if result.trim() != "0" && result.trim() != "CUDA_SUCCESS" {
                    return Err(format!(
                        "MPS did not confirm CUDA teardown for client {client}: {result}"
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    pub fn health(&mut self) -> Result<()> {
        if let Some(child) = &mut self.daemon
            && child.try_wait()?.is_some()
        {
            return Err("owned MPS service exited".into());
        }
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        if self.daemon.is_none() {
            return Ok(());
        }
        if !self.clients()?.is_empty() {
            return Err("MPS service still has clients; refusing to stop it".into());
        }
        self.command("quit")?;
        if let Some(child) = self.daemon.as_mut() {
            let deadline = Instant::now() + Duration::from_secs(5);
            while child.try_wait()?.is_none() {
                if Instant::now() >= deadline {
                    return Err("MPS service did not shut down".into());
                }
                thread::sleep(Duration::from_millis(20));
            }
        }
        self.daemon = None;
        Ok(())
    }
}

pub(super) fn uid() -> u32 {
    // SAFETY: geteuid has no arguments or memory effects.
    unsafe { libc::geteuid() }
}

fn numbers(text: &str) -> Vec<u32> {
    text.split_whitespace()
        .filter_map(|value| value.parse().ok())
        .collect()
}

fn select_gpu(selection: &str) -> Result<String> {
    if selection.trim().is_empty() {
        return Err("select a GPU index or UUID".into());
    }
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,uuid,name,compute_cap,driver_version",
            "--format=csv,noheader",
        ])
        .output()
        .map_err(|error| format!("NVIDIA GPU inspection is unavailable: {error}"))?;
    if !output.status.success() {
        return Err("nvidia-smi could not inspect NVIDIA GPUs".into());
    }
    let text = String::from_utf8(output.stdout)?;
    let entries: Vec<Vec<&str>> = text
        .lines()
        .map(|line| line.split(',').map(str::trim).collect())
        .collect();
    let visibility = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    let visible = visible_entries(&entries, visibility.as_deref());
    let entry = if let Ok(index) = selection.parse::<usize>() {
        visible.get(index).copied()
    } else {
        visible
            .iter()
            .copied()
            .find(|entry| entry[1].starts_with(selection))
    }
    .ok_or("selected GPU is not available within CUDA_VISIBLE_DEVICES")?;
    if entry[3].parse::<f64>().unwrap_or(0.0) < 7.0 {
        return Err("MPS manager requires a Volta-or-newer NVIDIA GPU".into());
    }
    if entry[4]
        .split('.')
        .next()
        .unwrap_or("0")
        .parse::<u32>()
        .unwrap_or(0)
        < 470
    {
        return Err("MPS client teardown requires NVIDIA driver 470 or newer".into());
    }
    Ok(entry[1].to_owned())
}

fn visible_entries<'a>(
    entries: &'a [Vec<&'a str>],
    visibility: Option<&str>,
) -> Vec<&'a Vec<&'a str>> {
    let Some(visibility) = visibility else {
        return entries.iter().filter(|entry| entry.len() >= 5).collect();
    };
    let mut visible = Vec::new();
    for token in visibility.split(',').map(str::trim) {
        let matches: Vec<_> = entries
            .iter()
            .filter(|entry| {
                entry.len() >= 5
                    && (entry[0] == token || !token.is_empty() && entry[1].starts_with(token))
            })
            .collect();
        if matches.len() != 1 {
            break;
        }
        if !visible
            .iter()
            .any(|entry: &&Vec<&str>| entry[1] == matches[0][1])
        {
            visible.push(matches[0]);
        }
    }
    visible
}

impl Drop for Mps {
    fn drop(&mut self) {
        // Failed startup must not leave an unused owned service behind. Never
        // shut down an attached service or a service with surviving clients.
        if self.daemon.is_some() && self.clients().is_ok_and(|clients| clients.is_empty()) {
            let _ = self.stop();
        }
    }
}

fn wait_output(mut child: Child, timeout: Duration) -> Result<Output> {
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err("MPS control command timed out".into());
        }
        thread::sleep(Duration::from_millis(10));
    };
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    if let Some(mut stream) = child.stdout.take() {
        stream.read_to_end(&mut stdout)?;
    }
    if let Some(mut stream) = child.stderr.take() {
        stream.read_to_end(&mut stderr)?;
    }
    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

pub(super) fn socket_directory() -> Result<PathBuf> {
    let path = std::env::temp_dir().join(format!("wf-manager-{}-{}", uid(), random_id()?));
    fs::create_dir(&path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_device_order_and_invalid_markers_match_cuda_semantics() {
        let entries = vec![
            vec!["0", "GPU-alpha", "gpu", "8.9", "580"],
            vec!["1", "GPU-beta", "gpu", "8.9", "580"],
        ];
        let visible = visible_entries(&entries, Some("1,0"));
        assert_eq!(
            visible.iter().map(|entry| entry[1]).collect::<Vec<_>>(),
            ["GPU-beta", "GPU-alpha"]
        );
        assert!(visible_entries(&entries, Some("")).is_empty());
        assert_eq!(visible_entries(&entries, Some("0,-1,1")).len(), 1);
        assert_eq!(
            visible_entries(&entries, Some("GPU-beta"))[0][1],
            "GPU-beta"
        );
    }
}
