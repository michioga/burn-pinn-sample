use clap::{Parser, Subcommand};

/// clapでコマンドラインの構造を定義します。
#[derive(Parser, Debug)]
#[command(author, version, about = "A Physics-Informed Neural Network (PINN) example with Burn", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// 実行するサブコマンドを定義します（train または infer）。
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// PINNモデルを学習し、結果をファイルに保存します
    Train,
    /// 保存されたPINNモデルを使い、推論を実行します
    Infer,
}
