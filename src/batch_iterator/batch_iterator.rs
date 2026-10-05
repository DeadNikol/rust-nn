//! pipasosia

use crate::tensor::tensor::{Tensor, TensorData};
use rand::rng;
use rand::seq::SliceRandom;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
pub enum DataSource {
    Tensor(Rc<RefCell<TensorData>>), // готовые данные
    Images(Vec<String>),             // пути к картинкам
}

impl DataSource {
    pub fn len(&self) -> usize {
        match self {
            DataSource::Tensor(t) => t.borrow().shape[0],
            DataSource::Images(paths) => paths.len(),
        }
    }

    pub fn sample_size(&self) -> usize {
        match self {
            DataSource::Tensor(t) => t.borrow().shape[1..].iter().product(),
            DataSource::Images(paths) => {
                if paths.is_empty() {
                    return 0;
                }

                let img = Tensor::from_path(&paths[0]).unwrap();
                img.tensor_data.borrow().shape[1..].iter().product()
            }
        }
    }

    pub fn sample_shape(&self) -> Vec<usize> {
        match self {
            DataSource::Tensor(t) => t.borrow().shape[1..].to_vec(),
            DataSource::Images(paths) => {
                if paths.is_empty() {
                    return vec![];
                }
                let img = Tensor::from_path(&paths[0]).unwrap();
                img.tensor_data.borrow().shape.clone()
            }
        }
    }

    pub fn get_sample(&self, idx: usize) -> Vec<f32> {
        match self {
            DataSource::Tensor(t) => {
                let data = t.borrow();
                let sample_size: usize = data.shape[1..].iter().product();
                let start = idx * sample_size;
                data.data[start..start + sample_size].to_vec()
            }
            DataSource::Images(paths) => {
                let img = Tensor::from_path(&paths[idx]).unwrap();
                img.tensor_data.borrow().data.clone()
            }
        }
    }
}

pub struct BatchIterator {
    pub(crate) x_source: DataSource,
    pub(crate) y_source: DataSource,
    pub(crate) batch_size: usize,
    pub(crate) indices: Vec<usize>,
    pub(crate) current: usize,
}

impl BatchIterator {
    pub fn new(x_source: DataSource, y_source: DataSource, batch_size: usize) -> Self {
        let num_samples = x_source.len();
        let indices: Vec<usize> = (0..num_samples).collect();
        Self {
            x_source,
            y_source,
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
        let num_samples = self.x_source.len();
        (num_samples + self.batch_size - 1) / self.batch_size
    }
}

impl Iterator for BatchIterator {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let num_samples = self.x_source.len();
        if self.current >= num_samples {
            return None;
        }

        let start = self.current;
        let end = (self.current + self.batch_size).min(num_samples);
        let batch_len = end - start;

        // Получаем размер одного образца
        let x_sample_size: usize = self.x_source.sample_size();
        let y_sample_size: usize = self.y_source.sample_size();

        let mut batch_x = Vec::with_capacity(batch_len * x_sample_size);
        let mut batch_y = Vec::with_capacity(batch_len * y_sample_size);

        // Собираем батч
        for &idx in &self.indices[start..end] {
            // X — из DataSource (лениво или из памяти)
            let x_sample = self.x_source.get_sample(idx);
            batch_x.extend(x_sample);

            // Y — из DataSource (унифицированно!)
            let y_sample = self.y_source.get_sample(idx);
            batch_y.extend(y_sample);
        }

        self.current = end;

        // Формируем форму для X (batch + остальные размерности)
        let x_shape = {
            let mut shape = vec![batch_len];
            shape.extend(self.x_source.sample_shape());
            shape
        };

        // Формируем форму для Y (batch + остальные размерности)
        let y_shape = {
            let mut shape = vec![batch_len];
            shape.extend(self.y_source.sample_shape());
            shape
        };

        let x_batch = Tensor::new(batch_x, x_shape, false, vec![], None);
        let y_batch = Tensor::new(batch_y, y_shape, false, vec![], None);

        Some((x_batch, y_batch))
    }
}
