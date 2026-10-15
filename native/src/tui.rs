//! Queue-first, attachable viewer using Workflow's terminal lease and palette.

use std::path::Path;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState, Wrap};
use serde_json::json;

use crate::manager::{Job, Parent, Status, discover, request, terminate_parent};
use scientific_workflow::external::{
    CommandInput, EditAction, TaskStatus, UsageMonitor, UsageSnapshot, interactive, open_surface,
    task_status_style,
};

#[derive(Default)]
struct View {
    level: u8,
    parent_id: Option<String>,
    job_id: Option<String>,
    parent_index: usize,
    job_index: usize,
    log_scroll: u16,
    stdout: bool,
    input: CommandInput,
    notice: String,
    quit: bool,
    terminating_parent: Option<String>,
}

pub(crate) fn run(root: &Path, pid: Option<u32>) -> Result<(), Box<dyn std::error::Error>> {
    if !interactive() {
        return Err("tui requires an interactive terminal; use screen or tmux".into());
    }
    let mut parents = discover(root);
    let mut view = View::default();
    if let Some(pid) = pid {
        let parent = parents
            .iter()
            .find(|parent| parent.pid == pid)
            .ok_or("parent PID was not found in this project")?;
        view.parent_id = Some(parent.id.clone());
        view.level = 1;
    }
    let (mut terminal, _lease) = open_surface()?;
    let mut usage = UsageMonitor::default();
    let mut refresh = Instant::now();
    let mut redraw = true;
    while !view.quit {
        if refresh.elapsed() >= Duration::from_millis(250) {
            parents = discover(root);
            refresh = Instant::now();
            redraw = true;
        }
        view.clamp(&parents);
        if redraw {
            let resources = usage.sample(Some(root));
            terminal.draw(|frame| render(frame, &parents, &view, resources))?;
            redraw = false;
        }
        if event::poll(Duration::from_millis(30))? {
            match event::read()? {
                Event::Key(key)
                    if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
                {
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('c')
                    {
                        view.quit = true;
                        continue;
                    }
                    view.key(key.code, &parents);
                    parents = discover(root);
                    redraw = true;
                }
                Event::Resize(_, _) => redraw = true,
                _ => {}
            }
        }
    }
    Ok(())
}

impl View {
    fn parent<'a>(&self, parents: &'a [Parent]) -> Option<&'a Parent> {
        if self.level == 0 {
            parents.get(self.parent_index)
        } else {
            parents
                .iter()
                .find(|parent| Some(&parent.id) == self.parent_id.as_ref())
        }
    }

    fn job<'a>(&self, parent: &'a Parent) -> Option<&'a Job> {
        if let Some(id) = &self.job_id {
            parent.jobs.iter().find(|job| &job.id == id)
        } else {
            ordered_jobs(parent).get(self.job_index).copied()
        }
    }

    fn clamp(&mut self, parents: &[Parent]) {
        if self
            .terminating_parent
            .as_ref()
            .is_some_and(|id| !parents.iter().any(|parent| &parent.id == id))
        {
            self.terminating_parent = None;
            self.notice = "Parent terminated and removed. Logs and outputs retained.".into();
        }
        if self.level > 0
            && !parents
                .iter()
                .any(|parent| Some(&parent.id) == self.parent_id.as_ref())
        {
            self.level = 0;
            self.parent_id = None;
            self.job_id = None;
        }
        self.parent_index = self.parent_index.min(parents.len().saturating_sub(1));
        if let Some(parent) = self.parent(parents) {
            let jobs = ordered_jobs(parent);
            if let Some(id) = &self.job_id
                && let Some(index) = jobs.iter().position(|job| &job.id == id)
            {
                self.job_index = index;
            }
            self.job_index = self.job_index.min(jobs.len().saturating_sub(1));
        }
    }

    fn key(&mut self, key: KeyCode, parents: &[Parent]) {
        if key == KeyCode::Enter && !self.input.text().is_empty() {
            let text = self.input.text().trim_start_matches(':').to_owned();
            self.input.edit_text(EditAction::Clear);
            if matches!(text.trim(), "q" | "quit" | "exit") {
                self.quit = true;
                return;
            }
            self.notice = match self.command(&text, parents) {
                Ok(()) => format!("{text}: accepted"),
                Err(error) => error.to_string(),
            };
            return;
        }
        let editing = !self.input.text().is_empty();
        match key {
            KeyCode::Esc => {
                if editing {
                    self.input.edit_text(EditAction::Clear);
                } else {
                    self.level = self.level.saturating_sub(1);
                }
            }
            KeyCode::Char('q') if !editing => self.quit = true,
            KeyCode::Char('t') if !editing => {
                if let Some(parent) = self.parent(parents) {
                    self.terminating_parent = Some(parent.id.clone());
                    self.notice = match terminate_parent(parent) {
                        Ok(()) => format!(
                            "Terminating '{}' and removing it after cleanup.",
                            parent.title
                        ),
                        Err(error) => error.to_string(),
                    };
                }
            }
            KeyCode::Left if !editing => self.level = self.level.saturating_sub(1),
            KeyCode::Right | KeyCode::Enter if !editing => {
                if self.level == 0 {
                    if let Some(parent) = parents.get(self.parent_index) {
                        self.parent_id = Some(parent.id.clone());
                        self.job_id = None;
                        self.job_index = 0;
                        self.level = 1;
                    }
                } else if self.level == 1 {
                    if let Some(parent) = self.parent(parents)
                        && let Some(job) = ordered_jobs(parent).get(self.job_index)
                    {
                        self.job_id = Some(job.id.clone());
                        self.level = 2;
                    }
                } else if self.level == 2 {
                    self.level = 3;
                    self.log_scroll = 0;
                }
            }
            KeyCode::Up | KeyCode::PageUp if !editing => {
                let step = if key == KeyCode::PageUp { 10 } else { 1 };
                if self.level == 0 {
                    self.parent_index = self.parent_index.saturating_sub(step);
                } else if self.level == 1 {
                    self.job_index = self.job_index.saturating_sub(step);
                    self.job_id = None;
                } else {
                    self.log_scroll = self.log_scroll.saturating_sub(step as u16);
                }
            }
            KeyCode::Down | KeyCode::PageDown if !editing => {
                let step = if key == KeyCode::PageDown { 10 } else { 1 };
                if self.level == 0 {
                    self.parent_index += step;
                } else if self.level == 1 {
                    self.job_index += step;
                    self.job_id = None;
                } else {
                    self.log_scroll = self.log_scroll.saturating_add(step as u16);
                }
            }
            KeyCode::Tab => {
                self.stdout = !self.stdout;
                self.log_scroll = 0;
            }
            KeyCode::Char(character) => {
                self.input.edit_text(EditAction::Insert(character));
            }
            KeyCode::Backspace => {
                self.input.edit_text(EditAction::Backspace);
            }
            KeyCode::Delete => {
                self.input.edit_text(EditAction::Delete);
            }
            KeyCode::Left => {
                self.input.edit_text(EditAction::Left);
            }
            KeyCode::Right => {
                self.input.edit_text(EditAction::Right);
            }
            KeyCode::Home => {
                self.input.edit_text(EditAction::Home);
            }
            KeyCode::End => {
                self.input.edit_text(EditAction::End);
            }
            _ => {}
        }
    }

    fn command(&self, text: &str, parents: &[Parent]) -> Result<(), Box<dyn std::error::Error>> {
        let parent = self.parent(parents).ok_or("no parent selected")?;
        let mut words = text.trim().splitn(2, char::is_whitespace);
        let action = words.next().unwrap_or("");
        let rest = words.next().unwrap_or("").trim();
        let selected = if self.level > 0 {
            self.job(parent).map(|job| job.id.clone())
        } else {
            None
        };
        let mut command = json!({"action": action, "job_id": selected});
        match action {
            "pause" | "resume" | "kill" | "remove" | "hold" | "release" => {
                if !rest.is_empty() { command["job_id"] = json!(rest); }
            }
            "delete" => { command["action"] = json!("remove"); }
            "pause-all" | "resume-all" => {
                command["action"] = json!(if action == "pause-all" { "pause" } else { "resume" }); command["job_id"] = json!(null);
            }
            "shuffle" | "shutdown" => { command["job_id"] = json!(null); }
            "title" => { command["job_id"] = json!(null); command["title"] = json!(rest); }
            "name" => { command["action"] = json!("title"); command["title"] = json!(rest); }
            "move" => { command["position"] = json!(rest.parse::<u64>().map_err(|_| "move requires a zero-based queue position")?); }
            "top" => { command["action"] = json!("move"); command["position"] = json!(0); }
            "bottom" => { command["action"] = json!("move"); command["position"] = json!(parent.pending.len()); }
            _ => return Err("commands: pause/resume, hold/release, kill, remove, move N, top/bottom, shuffle, title TEXT, shutdown; q detaches".into()),
        }
        request(parent, command)?;
        Ok(())
    }
}

fn ordered_jobs(parent: &Parent) -> Vec<&Job> {
    let mut jobs: Vec<&Job> = parent
        .jobs
        .iter()
        .filter(|job| job.state != Status::Removed)
        .collect();
    let positions: std::collections::HashMap<&str, usize> = parent
        .pending
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect();
    jobs.sort_by_cached_key(|job| {
        let group = if matches!(
            job.state,
            Status::Starting | Status::Running | Status::Suspended
        ) {
            0
        } else if job.state == Status::Pending {
            1
        } else {
            2
        };
        let position = positions
            .get(job.id.as_str())
            .copied()
            .unwrap_or(usize::MAX);
        (
            group,
            if group == 1 {
                position as u64
            } else {
                job.submitted_at
            },
        )
    });
    jobs
}

fn status_style(status: Status) -> Style {
    task_status_style(match status {
        Status::Pending => TaskStatus::Pending,
        Status::Starting | Status::Running => TaskStatus::Running,
        Status::Suspended | Status::Cancelled => TaskStatus::Cancelled,
        Status::Succeeded => TaskStatus::Completed,
        Status::Failed => TaskStatus::Failed,
        Status::Removed => TaskStatus::Skipped,
    })
}

fn render(frame: &mut ratatui::Frame<'_>, parents: &[Parent], view: &View, usage: UsageSnapshot) {
    let area = frame.area();
    let [heading, table, messages, footer] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(5),
        Constraint::Length(3),
        Constraint::Length(4),
    ])
    .areas(area);
    let title = if view.level == 0 {
        "MPS MANAGER · QUEUES".to_owned()
    } else {
        view.parent(parents)
            .map(|parent| format!("MPS MANAGER · {} · parent PID {}", parent.title, parent.pid))
            .unwrap_or_else(|| "MPS MANAGER".into())
    };
    frame.render_widget(
        Paragraph::new(title)
            .alignment(Alignment::Center)
            .style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .block(Block::default().borders(Borders::BOTTOM)),
        heading,
    );
    if view.level == 0 {
        render_parents(frame, table, parents, view.parent_index);
    } else if let Some(parent) = view.parent(parents) {
        if view.level == 1 {
            render_jobs(frame, table, parent, view.job_index);
        } else if let Some(job) = view.job(parent) {
            if view.level == 2 {
                let content = format!(
                    "Title: {}\nJob: {}\nPID{}: {}\nStatus: {}\nScript: {}\nArguments: {}\nFixed/common arguments: {}\nResolved parameters: {}\nGPU admission: {}\nWorking directory: {}\nOutput area: {}\nExit code: {}\n{}",
                    job.title,
                    job.id,
                    if job.state.terminal() {
                        " (exited)"
                    } else {
                        ""
                    },
                    job.pid.map_or("—".into(), |pid| pid.to_string()),
                    job.state.label(),
                    job.script_path.display(),
                    serde_json::to_string(&job.args).unwrap_or_default(),
                    serde_json::to_string(&job.common_args).unwrap_or_default(),
                    job.parameters
                        .as_ref()
                        .map_or_else(|| "—".into(), |v| v.to_string()),
                    job.admission_sequence
                        .map_or_else(|| "—".into(), |n| n.to_string()),
                    job.cwd.display(),
                    job.run_dir.display(),
                    job.exit_code.map_or("—".into(), |code| code.to_string()),
                    job.detail
                );
                frame.render_widget(
                    Paragraph::new(content)
                        .wrap(Wrap { trim: false })
                        .scroll((view.log_scroll, 0))
                        .block(
                            Block::default()
                                .borders(Borders::ALL)
                                .title(" Job details · → logs "),
                        ),
                    table,
                );
            } else {
                let stream = if view.stdout { "stdout" } else { "stderr" };
                let path = job
                    .run_dir
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join(format!("{stream}.log"));
                let content = crate::manager::tail(&path).unwrap_or_else(|error| error.to_string());
                frame.render_widget(
                    Paragraph::new(clean(&content))
                        .scroll((view.log_scroll, 0))
                        .block(
                            Block::default()
                                .borders(Borders::ALL)
                                .title(format!(" {stream} · Tab switches stream ")),
                        ),
                    table,
                );
            }
        }
    }
    let notice = if !view.notice.is_empty() {
        view.notice.clone()
    } else {
        view.parent(parents).and_then(|parent| parent.error.clone().or_else(|| parent.messages.last().cloned())).unwrap_or_else(|| "Enter a parent queue to manage its children. Closing this viewer leaves queues running.".into())
    };
    frame.render_widget(
        Paragraph::new(clean(&notice))
            .wrap(Wrap { trim: false })
            .block(Block::default().borders(Borders::ALL).title(" Messages ")),
        messages,
    );
    let percent = |value: Option<f64>| value.map_or("—".into(), |value| format!("{value:.0}%"));
    let help = if view.level == 0 {
        "↑/↓ select parent · Enter/→ open · q detach · t terminate parent · : commands"
    } else {
        "↑/↓ select · ← back · → details/logs · q detach · t terminate parent · : commands"
    };
    let lines = vec![
        Line::from(help),
        Line::from(vec![
            Span::styled("> ", Style::default().fg(Color::Cyan)),
            Span::raw(view.input.text()),
        ]),
        Line::from(format!(
            "CPU {}  RAM {}  DISK {} · No automatic retries",
            percent(usage.cpu_percent),
            percent(usage.ram_percent),
            percent(usage.disk_percent)
        )),
    ];
    frame.render_widget(Paragraph::new(lines), footer);
}

fn clean(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_control() || matches!(character, '\n' | '\t'))
        .collect()
}

fn render_parents(frame: &mut ratatui::Frame<'_>, area: Rect, parents: &[Parent], selected: usize) {
    let rows = parents.iter().map(|parent| {
        let (pending, live, succeeded, failed) = parent.counts();
        let state = if parent.alive() {
            if crate::manager::process_identity(parent.pid)
                .is_some_and(|(state, _, _)| state == 'T' || state == 't')
            {
                "suspended".into()
            } else if parent.state == "ready"
                && !parent.jobs.is_empty()
                && parent.jobs.iter().all(|job| job.state.terminal())
            {
                "finished".into()
            } else if parent.admission_paused {
                "paused".into()
            } else {
                parent.state.clone()
            }
        } else if parent.state == "closed" {
            "closed".into()
        } else {
            "offline".into()
        };
        Row::new(vec![
            parent.title.clone(),
            parent.pid.to_string(),
            if parent.mps {
                format!("GPU {}", parent.gpu)
            } else {
                "process".into()
            },
            state,
            live.to_string(),
            pending.to_string(),
            succeeded.to_string(),
            failed.to_string(),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Min(20),
            Constraint::Length(9),
            Constraint::Length(9),
            Constraint::Length(10),
            Constraint::Length(6),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(7),
        ],
    )
    .header(
        Row::new([
            "parent title",
            "PID",
            "device",
            "status",
            "live",
            "pending",
            "done",
            "failed",
        ])
        .style(Style::default().fg(Color::Cyan)),
    )
    .row_highlight_style(
        Style::default()
            .bg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    )
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Parent queues · Enter to inspect "),
    );
    let mut state = TableState::default().with_selected(if parents.is_empty() {
        None
    } else {
        Some(selected)
    });
    frame.render_stateful_widget(table, area, &mut state);
}

fn render_jobs(frame: &mut ratatui::Frame<'_>, area: Rect, parent: &Parent, selected: usize) {
    let jobs = ordered_jobs(parent);
    let rows = jobs.iter().map(|job| {
        let position = parent
            .pending
            .iter()
            .position(|id| id == &job.id)
            .map_or_else(
                || {
                    if job.detail == "held" {
                        "held".into()
                    } else {
                        "—".into()
                    }
                },
                |index| index.to_string(),
            );
        Row::new(vec![
            Cell::from(job.title.clone()),
            Cell::from(job.id.clone()),
            Cell::from(if job.state.terminal() {
                "—".into()
            } else {
                job.pid.map_or("—".into(), |pid| pid.to_string())
            }),
            Cell::from(job.state.label()).style(status_style(job.state)),
            Cell::from(position),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Min(20),
            Constraint::Length(22),
            Constraint::Length(9),
            Constraint::Length(12),
            Constraint::Length(7),
        ],
    )
    .header(
        Row::new(["job title", "job ID", "PID", "status", "queue #"])
            .style(Style::default().fg(Color::Cyan)),
    )
    .row_highlight_style(
        Style::default()
            .bg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    )
    .block(Block::default().borders(Borders::ALL).title(format!(
            " Jobs · {} live / {} parent limit · {}{} ",
            parent.counts().1,
            parent.max_concurrent,
            parent
                .gpu_max_concurrent
                .map_or_else(String::new, |limit| format!("shared GPU limit {limit} · ")),
            if parent.admission_paused {
                "admission paused"
            } else {
                "accepting submissions"
            }
        )));
    let mut state = TableState::default().with_selected(if jobs.is_empty() {
        None
    } else {
        Some(selected)
    });
    frame.render_stateful_widget(table, area, &mut state);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrows_navigate_without_turning_into_commands() {
        let mut view = View {
            level: 3,
            ..View::default()
        };
        view.key(KeyCode::Left, &[]);
        assert_eq!(view.level, 2);
        view.key(KeyCode::Right, &[]);
        assert_eq!(view.level, 3);
        view.key(KeyCode::Char('p'), &[]);
        view.key(KeyCode::Left, &[]);
        assert_eq!(view.level, 3);
        assert_eq!(view.input.text(), "p");
    }

    #[test]
    fn control_characters_are_not_executed_by_log_viewer() {
        assert_eq!(clean("hi\x1b[31m\nthere\x07"), "hi[31m\nthere");
    }
}
