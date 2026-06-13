use crate::tensor::Tensor;

pub trait Operation {
    fn forward(&mut self, data: Tensor) -> Tensor;
}
pub trait ActivationFunction {
    fn forward(&mut self, data: Tensor) -> Tensor;
}
pub trait LossFunction {
    fn forward(&mut self, pred_data: Tensor, true_data: Tensor) -> Tensor;
}

pub struct Linear {
    pub weights: Tensor, // Веса слоя
    pub biases: Tensor,  // Смещения слоя
}

impl Linear {
    pub fn new(input_features: usize, output_features: usize) -> Self {
        let weights = Tensor::uniform(0.0, 1.0, vec![1, input_features, output_features], true);
        let biases = Tensor::uniform(0.0, 1.0, vec![1, 1, output_features], true);

        Linear { weights, biases }
    }
}
impl Operation for Linear {
    fn forward(&mut self, data: Tensor) -> Tensor {
        let y = data.matmul(&self.weights).add(&self.biases);
        y
    }
}

pub struct ReLu {}
impl ReLu {
    pub fn new() -> Self {
        ReLu {}
    }
}
impl ActivationFunction for ReLu {
    fn forward(&mut self, data: Tensor) -> Tensor {
        data.relu()
    }
}

pub struct MSE {}
impl MSE {
    pub fn new() -> Self {
        MSE {}
    }
}
impl LossFunction for MSE {
    fn forward(&mut self, pred_data: Tensor, true_data: Tensor) -> Tensor {
        pred_data.mse(&true_data)
    }
}