mod manager;
mod tui;

fn main() {
    if let Err(error) = manager::run_cli() {
        eprintln!("mps-manager: {error}");
        std::process::exit(1);
    }
}
