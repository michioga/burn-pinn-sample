use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Tanh};
use burn::prelude::Backend;
use burn::tensor::Tensor;

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
