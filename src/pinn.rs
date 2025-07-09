use crate::model::Model;
use burn::nn::loss::{MseLoss, Reduction};
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

/// 物理損失を計算します。
///
/// 移流方程式の残差（方程式の各項を移項した結果、0になるべき値）を計算し、
/// その二乗平均誤差を損失として返します。
pub fn physics_loss<B: AutodiffBackend>(model: &Model<B>, coords: Tensor<B, 2>) -> Tensor<B, 1> {
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
