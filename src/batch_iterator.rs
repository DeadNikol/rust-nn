//! pipasosia

use crate::tensor::{Tensor, TensorData};
use rand::rng;
use rand::seq::SliceRandom;
use std::cell::RefCell;
use std::rc::Rc;

pub struct BatchIterator {
    pub x_data: Rc<RefCell<TensorData>>,
    pub targets: Rc<RefCell<TensorData>>,
    pub batch_size: usize,
    pub indices: Vec<usize>,
    pub current: usize,
}

impl BatchIterator {
    pub fn new(
        x_data: Rc<RefCell<TensorData>>,
        targets: Rc<RefCell<TensorData>>,
        batch_size: usize,
    ) -> Self {
        let num_samples = x_data.borrow().shape[0];
        let indices: Vec<usize> = (0..num_samples).collect();

        Self {
            x_data,
            targets,
            batch_size,
            indices,
            current: 0,
        }
    }

    pub fn reset_indices(&mut self) {
        let mut rng = rng();
        self.indices.shuffle(&mut rng);
        self.current = 0;
    }

    pub fn num_batches(&self) -> usize {
        let num_samples = self.x_data.borrow().shape[0];
        (num_samples + self.batch_size - 1) / self.batch_size
    }
}

impl Iterator for BatchIterator {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let num_samples = self.x_data.borrow().shape[0];
        let x_data_samples: usize = self.x_data.borrow().shape[1..].iter().product();
        let target_data_cols: usize = self.targets.borrow().shape[1..].iter().product();

        if self.current >= num_samples {
            return None;
        }

        let start = self.current;
        let end = (self.current + self.batch_size).min(num_samples);
        let batch_len = end - start;

        // Подсчитываем смещения в плоских массивах
        let x_data_ref = self.x_data.borrow();
        let y_data_ref = self.targets.borrow();

        let mut batch_x = Vec::with_capacity(batch_len * x_data_samples);
        let mut batch_y = Vec::with_capacity(batch_len * target_data_cols);

        for &idx in &self.indices[start..end] {
            let x_start = idx * x_data_samples;
            let y_start = idx * target_data_cols;

            batch_x.extend_from_slice(&x_data_ref.data[x_start..x_start + x_data_samples]);
            batch_y.extend_from_slice(&y_data_ref.data[y_start..y_start + target_data_cols]);
        }

        self.current = end;

        let x_batch = Tensor::new(
            batch_x,
            // vec![batch_len, x_data_samples],
            {
                let mut shape = vec![batch_len];
                shape.extend_from_slice(&self.x_data.borrow().shape[1..]);
                shape
            },
            false,
            vec![],
            None,
        );

        let y_batch = Tensor::new(
            batch_y,
            // vec![batch_len, target_data_cols],
            {
                let mut shape = vec![batch_len];
                shape.extend_from_slice(&self.targets.borrow().shape[1..]);
                shape
            },
            false,
            vec![],
            None,
        );

        Some((x_batch, y_batch))
    }
}
