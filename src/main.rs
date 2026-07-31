mod tensor;

use core::panic;
use std::{cell::RefCell, rc::Rc};

use itertools::{Itertools, Position::Last, iproduct};
use rayon::vec;
use tensor::*;

use crate::network::Network;

mod batch_iterator;
mod graph;
mod layers;
mod network;
mod optims;

#[cfg(test)]
mod unit_tests;

fn main() {
    let layer = tensor::Tensor::new(
        (0..3 * 3).map(|f| f as f32).collect(),
        vec![1, 1, 3, 3],
        false,
        vec![],
        None,
    );
    let kernel = Tensor::new(
        (0..2 * 2).map(|f| f as f32).collect(),
        vec![1, 1, 2, 2],
        false,
        vec![],
        None,
    );

    // let output = conv2d(&layer, &kernel, (1, 1));
    // conv2d_backward(&output);

    let a = Tensor::new(
        (0..3 * 3 * 4 * 3).map(|f| f as f32).collect(),
        vec![3, 3, 4, 3],
        false,
        vec![],
        None,
    );
    let b = Tensor::new(
        (0..2 * 3 * 2 * 2).map(|f| f as f32).collect(),
        vec![2, 3, 2, 2],
        false,
        vec![],
        None,
    );

    let output = conv2d(&a, &b, (1, 1));
    // println!("{}", output);
    conv2d_backward(&output);
}

fn conv2d(layer: &Tensor, kernel: &Tensor, stride: (usize, usize)) -> Tensor {
    println!("LOG: Не забудь учитывать паддинг при расчёте");

    // Блок im2col

    let layer_shape = layer.tensor_data.borrow().shape.clone();
    let layer_data = layer.tensor_data.borrow().data.clone();
    let kernel_shape = kernel.tensor_data.borrow().shape.clone();
    let kernel_data = kernel.tensor_data.borrow().data.clone();

    let x_iter =
        (layer_shape[layer_shape.len() - 2] - kernel_shape[kernel_shape.len() - 2]) / stride.0 + 1;
    let y_iter =
        (layer_shape[layer_shape.len() - 1] - kernel_shape[kernel_shape.len() - 1]) / stride.1 + 1;

    let mut layer_im_2_col_data: Vec<f32> = Vec::with_capacity(
        x_iter * y_iter * kernel_shape[2] * kernel_shape[3] * kernel_shape[1] * kernel_shape[0],
    );

    for image in 0..layer_shape[0] {
        for x in 0..x_iter {
            for y in 0..y_iter {
                for layer in 0..layer_shape[1] {
                    for local_x in 0..kernel_shape[2] {
                        for local_y in 0..kernel_shape[3] {
                            let global_index = image * layer_shape[1] * layer_shape[2] * layer_shape[3] // images
                                    + layer * layer_shape[2] * layer_shape[3] // layers
                                    + x * stride.0 * layer_shape[3] + local_x * layer_shape[3] // x
                                    + y * stride.1 + local_y; // y

                            layer_im_2_col_data.push(layer_data[global_index]);
                        }
                    }
                }
            }
        }
    }

    let temp_tensor = Tensor::new(
        layer_im_2_col_data,
        vec![
            layer_shape[0] * x_iter * y_iter,
            layer_shape[1] * kernel_shape[2] * kernel_shape[3],
        ],
        false,
        vec![],
        None,
    );

    // println!("{}", temp_tensor);

    let temp_kernel = Tensor::new(
        kernel_data,
        vec![
            kernel_shape[0],
            kernel_shape[2] * kernel_shape[3] * kernel_shape[1],
        ],
        true,
        vec![],
        None,
    )
    .transpose();
    temp_kernel.tensor_data.borrow_mut().require_grad = true;

    // print!("{}", temp_kernel);

    // Блок с вычислением

    let temp_output = temp_tensor.matmul(&temp_kernel);
    // println!("{}", temp_output);

    // Блок с col2im
    let temp_output_shape = temp_output.tensor_data.borrow().shape.clone();
    temp_output.tensor_data.borrow_mut().operation = Some(Operation::Conv2d(
        temp_output_shape[0],
        temp_output_shape[1],
        stride,
        (layer_shape[0], layer_shape[1], layer_shape[2], layer_shape[3], ),
        (kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3], ),
    ));
    temp_output.tensor_data.borrow_mut().shape =
        vec![layer_shape[0], kernel_shape[0], x_iter, y_iter];

    let output_data = temp_output.tensor_data.borrow().data.clone();
    let mut temp_output_data: Vec<f32> = Vec::with_capacity(output_data.capacity());
    for image in 0..layer_shape[0] {
        for y in 0..y_iter {
            for layer in 0..kernel_shape[0] {
                for x in 0..x_iter {
                    let local_index = image * kernel_shape[0] * x_iter * y_iter //images
                    + layer * x_iter * y_iter // layers
                    + x * y_iter // x
                    + y; //y
                    temp_output_data.push(output_data[local_index]);
                }
            }
        }
    }
    temp_output.tensor_data.borrow_mut().data = temp_output_data;
    temp_output.tensor_data.borrow_mut().parents.push(layer.tensor_data.clone());
    temp_output.tensor_data.borrow_mut().parents.push(kernel.tensor_data.clone());

    // println!("{}", temp_output);
    temp_output
}

fn conv2d_backward(conv2d_output: &Tensor) {
    let parent_layer_im2col = conv2d_output.tensor_data.borrow().parents[0].clone();
    let parent_kernel_im2col = conv2d_output.tensor_data.borrow().parents[1].clone();

    let parent_layer_orig = conv2d_output.tensor_data.borrow().parents[2].clone();

    // println!("{:?}, {:?}", parent_layer_im2col.borrow().data, parent_kernel_im2col.borrow().data);
    // println!("{:?}", parent_kernel_im2col.borrow().grad().data);

    conv2d_output.backward();

    let temp_layer_grad = Tensor::new(
        parent_layer_orig.borrow().grad().data.clone(),
        vec![3,3,4,3],
        false,
        vec![],
        None,
    );
    println!("{}", temp_layer_grad);
}
