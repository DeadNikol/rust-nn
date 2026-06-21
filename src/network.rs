//! Крейт с структурой Нейронной Сети, содержащей в себе параметры модели

use std::{cell::RefCell, rc::Rc};

use crate::{
    batch_iterator::{self, BatchIterator},
    optims::SGD,
    tensor::{Tensor, TensorData},
};

pub enum Loss {
    MSE,
    CrossEntropyWithSoftmax,
}

pub struct Network {
    pub parametres: Vec<Rc<RefCell<TensorData>>>,
}

impl Network {
    pub fn new() -> Self {
        Network { parametres: vec![] }
    }

    pub fn Linear(&mut self, n_input: usize, n_output: usize) -> (Tensor, Tensor) {
        let weights = Tensor::uniform(-1., 1., vec![n_input, n_output], true);
        let biases = Tensor::uniform(-1., 1., vec![1, n_output], true);
        self.parametres.push(weights.tensor_data.clone());
        self.parametres.push(biases.tensor_data.clone());

        return (weights, biases);
    }

    // Основной метод: пользователь передаёт замыкание, которое строит граф
    pub fn forward<F>(&self, data: &Tensor, forward_fn: F) -> Tensor
    where
        F: FnOnce(&Tensor) -> Tensor,
    {
        forward_fn(data)
    }

    pub fn fit<F>(
        &self,
        epochs: usize,
        lr: f32,
        data: Tensor,
        y_true: Tensor,
        loss_function: Loss,
        batch_size: usize,
        verbose: usize,
        forward_fn: F,
    ) where
        F: Fn(&Tensor) -> Tensor,
    {
        let mut optimizer = SGD::new(self.parametres.clone(), lr, 0.9);

        let mut losses: Vec<f32> = vec![];

        for epoch in 0..epochs {
            let mut batch_iterator = BatchIterator::new(
                data.tensor_data.clone(),
                y_true.tensor_data.clone(),
                batch_size,
            );
            let mut temp_losses_during_one_epoch: Vec<f32> = vec![];

            for (x, y) in batch_iterator.by_ref() {
                let pred = self.forward(&x, &forward_fn);
                let batch_loss = match loss_function {
                    Loss::MSE => pred.mse(&y),
                    Loss::CrossEntropyWithSoftmax => pred.cross_entropy_with_softmax(&y),
                };
                batch_loss.backward();
                optimizer.backward_step();

                temp_losses_during_one_epoch.push(batch_loss.tensor_data.borrow().data[0].clone());
                // optimizer.backward_step(&loss);
            }
            batch_iterator.reset_indices();
            // optimizer.backward_step();

            let total: f32 = temp_losses_during_one_epoch.iter().sum();
            let epoch_loss = total / temp_losses_during_one_epoch.len() as f32;
            losses.push(epoch_loss);
            if verbose != 0 && epoch % verbose == 0 {
                println!("epoch {}: loss - {}", epoch, epoch_loss);
            }
        }
        plot_losses(&losses, "loss_plot.png").unwrap();
    }
}

use plotters::prelude::*;
pub fn plot_losses(losses: &[f32], output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if losses.is_empty() {
        return Ok(());
    }

    // Находим минимальное и максимальное значение
    let min_loss = losses.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_loss = losses.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let padding = (max_loss - min_loss) * 0.1;

    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..losses.len(), (min_loss - padding)..(max_loss + padding))?;

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Loss")
        .draw()?;

    let data: Vec<(usize, f32)> = losses.iter().enumerate().map(|(i, &l)| (i, l)).collect();
    chart
        .draw_series(LineSeries::new(data, &RED))?
        .label("Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}
