//! # 物理情報ニューラルネットワーク (PINN) サンプルプログラム
//!
//! `burn` フレームワークを使用して、物理情報ニューラルネットワーク（PINN）を構築し、
//! 1次元の移流方程式を解くサンプルです。
//!
//! `clap` クレートを利用して、コマンドラインから`train`（学習）と`infer`（推論）の
//! 機能を個別に実行できます。
//!
//! ## 使い方
//!
//! ### 学習
//! ```bash
//! cargo run --release -- train
//! ```
//!
//! ### 推論
//! ```bash
//! cargo run --release -- infer
//! ```

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig, Tanh};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Distribution, Tensor};
use clap::{Parser, Subcommand};
use plotters::prelude::*;
use std::f32::consts::PI;
use std::path::Path;
use std::time::Instant;

/// モデルを保存するファイル名
const MODEL_FILENAME: &str = "pinn_model.mpk";

// --- clap: コマンドラインインターフェースの定義 ---

/// clapでコマンドラインの構造を定義します。
#[derive(Parser, Debug)]
#[command(author, version, about = "A Physics-Informed Neural Network (PINN) example with Burn", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// 実行するサブコマンドを定義します（train または infer）。
#[derive(Subcommand, Debug)]
enum Commands {
    /// PINNモデルを学習し、結果をファイルに保存します
    Train,
    /// 保存されたPINNモデルを使い、推論を実行します
    Infer,
}

// --- PINNモデルの定義 ---

/// PINNの本体となるニューラルネットワークモデル。
///
/// 座標(t, x)を入力とし、その点における物理量uを予測する多層パーセプトロン（MLP）です。
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linears: Vec<Linear<B>>,
    activation: Tanh,
}

impl<B: Backend> Model<B> {
    /// 新しいモデルを初期化します。
    pub fn new(device: &B::Device) -> Self {
        let n_hidden = 20;
        let n_layers = 4;
        let mut linears = Vec::new();
        linears.push(LinearConfig::new(2, n_hidden).init(device));
        for _ in 1..(n_layers - 1) {
            linears.push(LinearConfig::new(n_hidden, n_hidden).init(device));
        }
        linears.push(LinearConfig::new(n_hidden, 1).init(device));
        Self {
            linears,
            activation: Tanh::new(),
        }
    }

    /// モデルの順伝播を実行します。
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;
        for i in 0..(self.linears.len() - 1) {
            x = self.linears[i].forward(x);
            x = self.activation.forward(x);
        }
        self.linears.last().unwrap().forward(x)
    }
}

// --- PINNのロジック ---

/// 物理損失を計算します。
///
/// 移流方程式の残差（方程式の各項を移項した結果、0になるべき値）を計算し、
/// その二乗平均誤差を損失として返します。
fn physics_loss<B: AutodiffBackend>(model: &Model<B>, coords: Tensor<B, 2>) -> Tensor<B, 1> {
    let advection_speed = 1.0;
    let coords_grad = coords.clone().require_grad();
    let u = model.forward(coords_grad.clone());
    let grads_1 = u.clone().sum().backward();
    let u_grads = coords_grad.grad(&grads_1).unwrap();
    let u_t_inner = u_grads.clone().slice([0..coords.dims()[0], 0..1]);
    let u_x_inner = u_grads.slice([0..coords.dims()[0], 1..2]);
    let residual_inner = u_t_inner + u_x_inner.mul_scalar(advection_speed);
    let residual = Tensor::<B, 2>::from_inner(residual_inner);
    MseLoss::new().forward(residual, Tensor::zeros_like(&u), Reduction::Mean)
}

/// 学習過程の損失をグラフとしてPNGファイルに出力します。
fn plot_loss_history(
    total_loss_hist: &[f32],
    phys_loss_hist: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("loss_graph.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let max_log_loss = total_loss_hist.get(0).unwrap_or(&1.0).log10();
    let min_log_loss = total_loss_hist.last().unwrap_or(&1e-6).log10() - 0.5;
    let mut chart = ChartBuilder::on(&root)
        .caption("Loss History", ("sans-serif", 40).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..total_loss_hist.len(), min_log_loss..max_log_loss)?;
    chart
        .configure_mesh()
        .y_desc("Loss (log10 scale)")
        .x_desc("Epochs (x200)")
        .draw()?;
    chart
        .draw_series(LineSeries::new(
            total_loss_hist
                .iter()
                .enumerate()
                .map(|(i, &val)| (i, val.log10())),
            &RED,
        ))?
        .label("Total Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    chart
        .draw_series(LineSeries::new(
            phys_loss_hist
                .iter()
                .enumerate()
                .map(|(i, &val)| (i, val.log10())),
            &BLUE,
        ))?
        .label("Physics Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}

// --- サブコマンドに対応する関数 ---

/// `train`サブコマンドを実行します。
///
/// モデルの学習、損失グラフの描画、学習済みモデルのファイル保存を行います。
fn run_training() {
    type MyBackend = Autodiff<NdArray<f32>>;
    let device = Default::default();

    let n_ic = 100;
    let n_bc = 100;
    let n_collocation = 5000;
    let t_ic = Tensor::<MyBackend, 2>::zeros([n_ic, 1], &device);
    let x_ic = Tensor::<MyBackend, 2>::random([n_ic, 1], Distribution::Uniform(-1.0, 1.0), &device);
    let u_ic = x_ic.clone().mul_scalar(PI).sin();
    let coords_ic = Tensor::cat(vec![t_ic, x_ic], 1);
    let t_bc = Tensor::<MyBackend, 2>::random([n_bc, 1], Distribution::Uniform(0.0, 1.0), &device);
    let x_bc_neg1 = Tensor::ones_like(&t_bc).mul_scalar(-1.0);
    let x_bc_pos1 = Tensor::ones_like(&t_bc);
    let coords_bc_neg1 = Tensor::cat(vec![t_bc.clone(), x_bc_neg1], 1);
    let coords_bc_pos1 = Tensor::cat(vec![t_bc, x_bc_pos1], 1);
    let t_col = Tensor::<MyBackend, 2>::random(
        [n_collocation, 1],
        Distribution::Uniform(0.0, 1.0),
        &device,
    );
    let x_col = Tensor::<MyBackend, 2>::random(
        [n_collocation, 1],
        Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let collocation_coords = Tensor::cat(vec![t_col, x_col], 1);

    let mut model = Model::<MyBackend>::new(&device);
    let learning_rate = 1e-3;
    let mut optim = AdamConfig::new().init();

    let mut total_loss_history = Vec::new();
    let mut phys_loss_history = Vec::new();
    let training_start = Instant::now();

    println!("学習を開始します (移流方程式) - バックエンド: NdArray (CPU)");
    for epoch in 1..=8000 {
        let pred_ic = model.forward(coords_ic.clone());
        let loss_ic = MseLoss::new().forward(pred_ic, u_ic.clone(), Reduction::Mean);
        let pred_bc_neg1 = model.forward(coords_bc_neg1.clone());
        let pred_bc_pos1 = model.forward(coords_bc_pos1.clone());
        let loss_bc = MseLoss::new().forward(pred_bc_neg1, pred_bc_pos1, Reduction::Mean);
        let loss_phys = physics_loss(&model, collocation_coords.clone());
        let total_loss = loss_ic + loss_bc + loss_phys.clone();

        if epoch % 200 == 0 {
            let total_loss_val = total_loss.clone().into_scalar();
            let phys_loss_val = loss_phys.clone().into_scalar();
            total_loss_history.push(total_loss_val);
            phys_loss_history.push(phys_loss_val);
            println!(
                "[Epoch {}] Total Loss: {:.6}, Physics Loss: {:.6}",
                epoch, total_loss_val, phys_loss_val
            );
        }

        let grads = total_loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(learning_rate, model, grads);
    }
    let training_duration = training_start.elapsed();
    println!("学習が完了しました。");
    println!("=> 学習時間: {:.2?}", training_duration);

    if let Err(e) = plot_loss_history(&total_loss_history, &phys_loss_history) {
        eprintln!("グラフの描画に失敗しました: {}", e);
    } else {
        println!("=> 損失グラフを 'loss_graph.png' に保存しました。");
    }

    println!("学習済みモデルを保存中...");
    model
        .save_file(
            MODEL_FILENAME,
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        )
        .expect("モデルの保存に失敗しました。");
    println!("=> モデルを '{}' に保存しました。", MODEL_FILENAME);
}

/// `infer`サブコマンドを実行します。
///
/// ファイルから学習済みモデルを読み込み、推論を実行します。
fn run_inference() {
    type MyBackend = NdArray<f32>;
    let device = Default::default();

    if !Path::new(MODEL_FILENAME).exists() {
        eprintln!(
            "エラー: モデルファイル '{}' が見つかりません。",
            MODEL_FILENAME
        );
        eprintln!("最初に 'train' コマンドでモデルを学習・保存してください。");
        std::process::exit(1);
    }

    println!("\n推論を実行します - バックエンド: NdArray (CPU)");
    let inference_start = Instant::now();

    println!("保存済みモデルを '{}' からロード中...", MODEL_FILENAME);
    let model = Model::<MyBackend>::new(&device)
        .load_file(
            MODEL_FILENAME,
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            &device,
        )
        .expect("モデルの読み込みに失敗しました。");

    let n_t = 50;
    let n_x = 50;
    let t_infer_vals = (0..n_t)
        .map(|i| i as f32 / (n_t - 1) as f32)
        .collect::<Vec<f32>>();
    let x_infer_vals = (0..n_x)
        .map(|i| -1.0 + i as f32 * 2.0 / (n_x - 1) as f32)
        .collect::<Vec<f32>>();
    let mut infer_coords_vec = Vec::with_capacity(n_t * n_x * 2);
    for t_val in &t_infer_vals {
        for x_val in &x_infer_vals {
            infer_coords_vec.push(*t_val);
            infer_coords_vec.push(*x_val);
        }
    }

    let infer_coords_1d = Tensor::<MyBackend, 1>::from_floats(infer_coords_vec.as_slice(), &device);
    let infer_coords = infer_coords_1d.reshape([n_t * n_x, 2]);
    let predictions = model.forward(infer_coords);
    let inference_duration = inference_start.elapsed();

    println!(
        "推論が完了しました。入力グリッド数: {}x{}={}, 出力テンソルの形状: {:?}",
        n_t,
        n_x,
        n_t * n_x,
        predictions.dims()
    );
    println!("=> 推論時間: {:.2?}", inference_duration);
}

/// プログラムのエントリーポイント。
///
/// コマンドライン引数を解析し、`train`または`infer`の処理に振り分けます。
fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Train => {
            run_training();
        }
        Commands::Infer => {
            run_inference();
        }
    }
}
