//! Крейт с оптимизаторами

use std::cell::RefCell;
use std::rc::Rc;

use crate::tensor::*;

pub struct SGD {
    pub parametres: Vec<Rc<RefCell<TensorData>>>,
    pub lr: f32,
    pub momentum: f32,
    velosity: Vec<Vec<f32>>,
}

impl SGD {
    pub fn new(parametres: Vec<Rc<RefCell<TensorData>>>, lr: f32, momentum: f32) -> Self {
        let mut velosity: Vec<Vec<f32>> = vec![];
        for i in parametres.iter() {
            let node_data_len = i.borrow().data.len();
            velosity.push(vec![0.0; node_data_len]);
        }
        Self {
            parametres,
            lr,
            momentum,
            velosity,
        }
    }

    pub fn backward_step(&mut self) {
        // data.backward();
        for (index, i) in self.parametres.iter().enumerate() {
            let mut node = i.borrow_mut();
            // println!("{:?}", self.velosity[index]);

            for (index_of_value_in_vector, velosity_value) in
                self.velosity[index].iter_mut().enumerate()
            {
                *velosity_value =
                    *velosity_value * self.momentum + node.grad[index_of_value_in_vector] * self.lr;
                node.data[index_of_value_in_vector] -= *velosity_value;
            }
            node.zero_grad();
        }
    }
}
