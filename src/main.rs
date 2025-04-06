use clap::Parser;

use amaruensis::consts::*;

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    source: Option<String>,
    #[arg(long)]
    scanned: Option<String>,
}

fn main() {
    let cli = Cli::parse();
    amaruensis::do_main(
        cli.source.unwrap_or(EXAMPLE_PAPER_PATH.to_string()),
        cli.scanned.unwrap_or(EXAMPLE_SCANNED_PATH.to_string()),
    );
}
