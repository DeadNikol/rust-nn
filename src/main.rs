mod tensor;

use core::panic;
use std::{cell::RefCell, process::Output, rc::Rc, time::Instant};

use itertools::{Itertools, Position::Last, iproduct, max};
use plotters::style::TextStyle;
use rand::RngExt;
use rayon::{iter::ParallelIterator, slice::ParallelSlice, vec, yield_local};
use tensor::*;

use crate::network::{Loss, Network};

mod batch_iterator;
mod graph;
mod layers;
mod network;
mod optims;

#[cfg(test)]
mod unit_tests;

fn main() {
    // test_conv();
    // test_pooling();
    // test_conv_network();
    // TODO: сделать тесты на пулинг, на свёртку, на производные для всего этого. Возможно на reshape и flatten.
}

fn test_pooling() {
    let x = Tensor::new(
        (0..2 * 2 * 3 * 3).map(|f| f as f32).collect(),
        vec![2, 2, 3, 3],
        true,
        vec![],
        None,
    );

    println!("{}", x);
    // let output = max_pool(x.tensor_data.clone(), (2, 2), (1, 1));
    let output = x.max_pool((2, 2), (1, 1));
    println!("{}", output);
    output.backward();
    println!("{}", x.grad());
}

fn test_conv() {
    let test_image = Tensor::new(
        (0..1 * 3 * 4 * 3).map(|f| f as f32).collect(),
        vec![1, 3, 4, 3],
        true,
        vec![],
        None,
    );
    let kernel_1 = Tensor::new(
        (0..1 * 3 * 4 * 3).map(|f| f as f32).collect(),
        vec![1, 3, 4, 3],
        true,
        vec![],
        None,
    );

    // let output = conv2d(&a, &b, (1, 1));
    let output = test_image.conv2d(&kernel_1, (1, 1));
    println!("{}", output);
    output.backward();
    println!("{}", test_image.grad());
}

fn test_conv_network() {
    let mut net = network::Network::new();
    let kernel1 = net.Conv2d(3, 2, (3, 3));
    let kernel2 = net.Conv2d(2, 1, (3, 3));
    let (w1, b1) = net.Linear(1, 1);

    let x = Tensor::new(
        (0..2 * 3 * 10 * 10).map(|f| f as f32).collect(),
        vec![2, 3, 10, 10],
        false,
        vec![],
        None,
    );
    let y = Tensor::new(vec![0., 1.], vec![2, 1], false, vec![], None);
    let forward_fn = |x: &Tensor| {
        x.conv2d(&kernel1, (1, 1))
            .max_pool((2, 2), (2, 2))
            .relu()
            .conv2d(&kernel2, (1, 1))
            .max_pool((2, 2), (2, 2))
            .relu()
            .flatten()
            .matmul(&w1)
            .add(&b1)
    };

    let start = Instant::now();
    net.fit(
        1_000,
        0.01,
        x.clone(),
        y.clone(),
        Loss::MSE,
        20,
        100,
        forward_fn,
    );
    let duration = start.elapsed();

    println!("\n=== Training Time ===");
    println!("Duration: {:?}", duration);

    // Вычисляем финальную MSE
    let output = forward_fn(&x);
    let final_loss = output.mse(&y);
    // println!("Final MSE: {}", final_loss.tensor_data.borrow().data[0]);
    // assert!(
    //     final_loss.tensor_data.borrow().data[0] <= 0.02,
    //     "MSE {} больше приемлемого значения 0.02",
    //     final_loss.tensor_data.borrow().data[0]
    // );

    // Вычисляем точность
    let predictions: Vec<f32> = output.tensor_data.borrow().data.clone();
    let accuracy: f32 = predictions
        .iter()
        .zip(y.tensor_data.borrow().data.iter())
        .map(|(&pred, &target)| {
            let pred_binary = if pred > 0.5 { 1.0 } else { 0.0 };
            if pred_binary == target { 1.0 } else { 0.0 }
        })
        .sum::<f32>()
        / predictions.len() as f32;
    // assert!(
    //     accuracy >= 0.98,
    //     "Точность {} меньше приемлемого значения 0.98",
    //     accuracy
    // );

    println!("Accuracy: {:.4}", accuracy);
}
