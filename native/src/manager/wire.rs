use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

use serde_json::{Value, json};

use super::model::{Parent, Result};

pub(super) const FRAME_LIMIT: u64 = 8 * 1024 * 1024;

pub(crate) fn request(parent: &Parent, mut value: Value) -> Result<Value> {
    if !parent.alive() {
        return Err("parent is not running on this host".into());
    }
    if super::model::process_identity(parent.pid)
        .is_some_and(|(state, _, _)| state == 'T' || state == 't')
    {
        if value["action"] == "resume" && value["job_id"].is_null() {
            wake_parent(parent)?;
        } else {
            return Err("parent process is suspended; use resume on the parent first".into());
        }
    }
    value["queue_id"] = json!(parent.id);
    let mut stream = UnixStream::connect(&parent.socket)?;
    stream.set_read_timeout(Some(Duration::from_secs(20)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    send(&mut stream, &value)?;
    let reply = receive(&stream)?;
    if reply["ok"] != true {
        return Err(reply["error"]
            .as_str()
            .unwrap_or("manager request failed")
            .into());
    }
    Ok(reply["result"].clone())
}

pub(crate) fn wake_parent(parent: &Parent) -> Result<()> {
    if !parent.alive() {
        return Err("parent identity changed before resume".into());
    }
    use std::os::fd::{AsRawFd as _, FromRawFd as _};
    // SAFETY: pidfd identifies this captured CPU-only manager, never an MPS server.
    let raw = unsafe { libc::syscall(libc::SYS_pidfd_open, parent.pid, 0) } as i32;
    if raw < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let fd = unsafe { std::os::fd::OwnedFd::from_raw_fd(raw) };
    if !parent.alive() {
        return Err("parent identity changed before resume".into());
    }
    let result = unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            fd.as_raw_fd(),
            libc::SIGCONT,
            std::ptr::null::<libc::siginfo_t>(),
            0,
        )
    };
    if result != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

pub(crate) fn terminate_parent(parent: &Parent) -> Result<()> {
    if parent.alive() {
        request(parent, json!({"action": "terminate"}))?;
    } else {
        if !parent.jobs.iter().all(|job| job.state.terminal()) {
            return Err(
                "offline parent has unfinished jobs; inspect their PIDs before removing it".into(),
            );
        }
        let mut saved: Parent = super::model::read_json(&parent.root.join("state.json"))?;
        if saved.id != parent.id || saved.alive() {
            return Err("parent identity changed; refresh the viewer".into());
        }
        saved.dismissed = true;
        super::model::write_json(&parent.root.join("state.json"), &saved)?;
    }
    Ok(())
}

pub(super) fn receive(stream: &UnixStream) -> Result<Value> {
    use std::io::Read as _;
    let mut reader = BufReader::new(stream).take(FRAME_LIMIT + 1);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    if line.len() as u64 > FRAME_LIMIT || !line.ends_with('\n') {
        return Err("invalid or oversized manager message".into());
    }
    Ok(serde_json::from_str(&line)?)
}

pub(super) fn send(stream: &mut UnixStream, value: &Value) -> Result<()> {
    let data = serde_json::to_vec(value)?;
    if data.len() as u64 > FRAME_LIMIT {
        return Err("manager message is too large".into());
    }
    stream.write_all(&data)?;
    stream.write_all(b"\n")?;
    Ok(())
}

pub(super) fn same_user(stream: &UnixStream) -> bool {
    use std::os::fd::AsRawFd as _;
    let mut credentials = libc::ucred {
        pid: 0,
        uid: 0,
        gid: 0,
    };
    let mut size = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: SO_PEERCRED writes a ucred into the correctly sized buffer.
    unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&mut credentials as *mut libc::ucred).cast(),
            &mut size,
        ) == 0
            && credentials.uid == libc::geteuid()
    }
}
