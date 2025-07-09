use crate::MODEL_FILENAME;
use crate::model::Model;
use crate::pinn::physics_loss;
use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::{Distribution, Tensor};
use plotters::prelude::*;
use std::f32::consts::PI;
use std::time::Instant;

type MyBackend = Autodiff<NdArray<f32>>;

/// `train`サブコマンドを実行します。
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();

    // --- データセットの準備 ---
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

    // --- モデルとオプティマイザの初期化 ---
    let mut model = Model::<MyBackend>::new(&device);
    let mut optim = AdamConfig::new().init();
    let learning_rate = 1e-3;

    let mut total_loss_history = Vec::new();
    let mut phys_loss_history = Vec::new();
    let training_start = Instant::now();

    println!("学習を開始します (移流方程式) - バックエンド: NdArray (CPU)");

    // --- 学習ループ ---
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

    // --- 結果の保存と描画 ---
    plot_loss_history(&total_loss_history, &phys_loss_history)?;
    println!("=> 損失グラフを 'loss_graph.png' に保存しました。");

    println!("学習済みモデルを保存中...");
    match model.save_file(
        MODEL_FILENAME,
        &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
    ) {
        Ok(_) => (),
        Err(e) => return Err(Box::new(e)),
    };
    println!("=> モデルを '{}' に保存しました。", MODEL_FILENAME);

    Ok(())
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
