//! # 物理情報ニューラルネットワーク (PINN) ライブラリ
//!
//! `burn` フレームワークを使用して、物理情報ニューラルネットワーク（PINN）を構築し、
//! 1次元の移流方程式を解くための主要なコンポーネントを提供します。

pub mod cli;
pub mod inference;
pub mod model;
pub mod pinn;
pub mod training;

/// モデルを保存するファイル名
pub const MODEL_FILENAME: &str = "pinn_model.mpk";
