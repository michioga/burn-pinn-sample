use crate::MODEL_FILENAME;
use crate::model::Model;
use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::Tensor;
use std::path::Path;
use std::time::Instant;

type MyBackend = NdArray<f32>;

/// `infer`サブコマンドを実行します。
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();

    if !Path::new(MODEL_FILENAME).exists() {
        return Err(format!(
            "モデルファイル '{}' が見つかりません。\n最初に 'train' コマンドでモデルを学習・保存してください。",
            MODEL_FILENAME
        ).into());
    }

    println!("\n推論を実行します - バックエンド: NdArray (CPU)");
    let inference_start = Instant::now();

    println!("保存済みモデルを '{}' からロード中...", MODEL_FILENAME);
    let model = match Model::<MyBackend>::new(&device).load_file(
        MODEL_FILENAME,
        &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        &device,
    ) {
        Ok(loaded_model) => loaded_model,
        Err(e) => return Err(Box::new(e)),
    };

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

    Ok(())
}
