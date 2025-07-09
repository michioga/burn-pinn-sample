use clap::Parser;
use pinn::cli::{Cli, Commands};

/// プログラムのエントリーポイント。
///
/// コマンドライン引数を解析し、`train`または`infer`の処理に振り分けます。
fn main() {
    let cli = Cli::parse();

    let result = match &cli.command {
        Commands::Train => pinn::training::run(),
        Commands::Infer => pinn::inference::run(),
    };

    if let Err(e) = result {
        eprintln!("\nエラーが発生しました: {}", e);
        std::process::exit(1);
    }
}
