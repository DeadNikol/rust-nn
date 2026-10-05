//! В этом крейте прописаны классы Тензора, данных тензора, функции над тензорами, операция взятия производной, перечисление операций

// use crate::{
//     addictive_functions::{self, col2im, im2col}, graph,
// };
use super::addictive_functions::{col2im, im2col};
use crate::graph::graph;
use image::{DynamicImage, GenericImageView, ImageReader};
use rand::RngExt;
use rayon::prelude::*;
use std::{
    cell::RefCell,
    fmt::{self},
    path::Path,
    rc::Rc,
};

/// Перечисляет операции, которые могли породить тензор
#[derive(Debug)]
pub enum Operation {
    Matmul,
    Add,
    ReLu,
    MSE,
    Sigmoid,
    Tanh,
    CrossEntropyWithSoftmax,
    /// Содержит в себе stride (x, y) и
    /// Результат matmul после im2col
    Conv2d(
        (usize, usize), //stride
        Tensor,         // результат матмула после im2col
    ),
    /// Содержит в себе значения добавлений: `[слева, справа, сверху, снизу]`
    Padding(usize, usize, usize, usize),
    Reshape,
    /// Содержит в себе значения `(kernel_shape.x, kernel_shape.y)`, `(stride.x, stride.y)`
    MaxPool((usize, usize), (usize, usize)),
}

/// Внутренние данные тензора. Хранятся как Rc<RefCell<TensorData>>>
#[derive(Debug)]
pub struct TensorData {
    /// Сами данные тензора
    pub data: Vec<f32>, // Само значение тензора. Оно будет плоским, обращение к строкам и тому подобное будет только через индексы. Срезов, как numpy пока не планируется
    /// Размерность тензора. Пока что она обязательно должна быть трёхмерной
    pub shape: Vec<usize>, // Размерность тензора. Она должна быть строго трёхмерной
    /// Смещения для того, что бы можно было избежать строгой трёхмерности и быстро получать доступ к элементу по индексам
    pub stride: Vec<usize>,
    /// Флаг для обозначения необходимости взятия производной и построения графа вычислений дальше
    pub require_grad: bool, // Флаг необходимости взятия производной. Если true, то в память записываем все операции над этим тензором. Потом проходим в обратном порядке, и если этот тензор непосредственно участвовал в операции, то прибавляем его к значению градиента
    /// Список родителей, породивших этот тензор. В зависимости от операции, их может быть сколько угодно
    pub parents: Vec<Rc<RefCell<TensorData>>>, // Вектор ссылок на родителей переменной, по которым нужна переменная. Когда мы нажмём Tensor::backward(), если require_grad = false, то оно вызовет backward У родителей, где require_grad = true
    /// Значение градиента. Все операции прописаны для `data`, так что был прописан метод `grad()` для создания тензора с градиентов в дате.
    pub grad: Vec<f32>, // Значние градиента. Оно такой-же размерности, как и self.data. По умолчанию равны нулям
    /// Операция, которая породила тензор.
    pub operation: Option<Operation>, // Это операция, которая породила данный тензор. Если его только создали, то у него ни родителей, ни такой операции
}

impl TensorData {
    pub fn new(
        data: Vec<f32>,
        shape: Vec<usize>,
        require_grad: bool,
        parents: Vec<Rc<RefCell<TensorData>>>,
        operation: Option<Operation>,
    ) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "Невозможно конвертировать массив длины {} в тензор размерности {:?} (Ожидается массив длины {})",
            data.len(),
            shape,
            shape.iter().product::<usize>()
        );
        let mut stride: Vec<usize> = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        Self {
            data,
            shape,
            stride,
            require_grad,
            parents,
            grad: vec![],
            operation: operation,
        }
    }

    #[allow(dead_code)]
    pub fn zero_grad(&mut self) {
        for g in &mut self.grad {
            *g = 0.0
        }
    }
    /// Векторное умножение данных двух тензоров
    pub fn matmul(&self, right_tensor: &TensorData) -> Vec<f32> {
        assert!(
            self.shape.len() == 2 && right_tensor.shape.len() == 2,
            "Матричное перемножение доступно только для матриц (Всего 2 размерности), даны {}-мерный и {}-мерный тензоры.",
            self.shape.len(),
            right_tensor.shape.len()
        );

        let self_rows = self.shape[self.shape.len() - 2];
        let self_cols = self.shape[self.shape.len() - 1];
        let right_rows = right_tensor.shape[right_tensor.shape.len() - 2];
        let right_cols = right_tensor.shape[right_tensor.shape.len() - 1];

        assert_eq!(
            self_cols, right_rows,
            "Что бы векторно перемножить две матрицы, их размерности должны быть [x,n] @ [n,y] => [x,y]. Даны [{},{}] @ [{},{}] => ошибка",
            self_rows, self_cols, right_rows, right_cols
        );

        let m = self_rows;
        let k = self_cols;
        let n = right_cols;

        let a = &self.data;
        let b = &right_tensor.data;

        // Параллельная обработка строк

        let mut output_data = vec![0.0; m * n];
        output_data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, row)| {
                for l in 0..k {
                    let a_il = a[i * k + l];
                    let b_row = &b[l * n..(l + 1) * n];
                    for j in 0..n {
                        row[j] += a_il * b_row[j];
                    }
                }
            });
        output_data
    }

    pub fn add(&self, right_tensor: &TensorData) -> Vec<f32> {
        // Блок проверок
        assert_eq!(
            self.shape.len(),
            right_tensor.shape.len(),
            "Количество размерностей должно совпадать: {:?} vs {:?}",
            self.shape,
            right_tensor.shape
        );

        let mut difference: bool = false;
        let mut ability_to_add_right_to_left_flag: bool = true;
        let mut different_batch_index: usize = 0;
        // Проверка: если размерности совпадают или если не совпадает только одна размерность, а все большие == единице
        for i in (0..self.shape.len()).rev() {
            let dim_self = self.shape[i];
            let dim_right = right_tensor.shape[i];

            if dim_self == dim_right && !difference {
                continue;
            } else if dim_self != dim_right && dim_self > dim_right && !difference {
                difference = true;
                different_batch_index = i; // Запоминаем индекс различающегося батча
            } else if dim_self == 1 && dim_right == 1 && difference {
                continue;
            } else {
                ability_to_add_right_to_left_flag = false;
            }
        }
        assert!(
            ability_to_add_right_to_left_flag,
            "Нельзя слогать тензоры размерностей {:?} и {:?}. Отличаться должен только одна размерность, а все значения слева от неё должны быть равными единице",
            self.shape, right_tensor.shape
        );

        let temp_data = self.data.clone();

        if !difference {
            let temp_right_data = right_tensor.data.clone(); // Временный держатель данных правого операнда различается при бродкасте

            // Случай 1: одинаковые формы → поэлементное сложение
            let mut new_data = vec![0.0; self.data.len()];
            new_data.par_iter_mut().enumerate().for_each(|(i, v)| {
                *v = temp_data[i] + temp_right_data[i];
            });
            new_data
        } else {
            let temp_right_data = right_tensor
                .data
                .clone()
                .repeat(self.shape[different_batch_index]);

            let mut new_data = vec![0.0; self.data.len()];
            new_data.par_iter_mut().enumerate().for_each(|(i, v)| {
                *v = temp_data[i] + temp_right_data[i];
            });

            new_data

            // Случай 2: различается batch → broadcasting

            // let batch_self = self.shape[0];
            // let batch_right = right_tensor.shape[0];
            // let elements_per_batch: usize = self.shape[1..].iter().product();

            // let output_batch = batch_self.max(batch_right);
            // let mut new_data = vec![0.0; output_batch * elements_per_batch];

            // new_data
            //     .par_chunks_mut(elements_per_batch)
            //     .enumerate()
            //     .for_each(|(batch, chunk)| {
            //         for i in 0..elements_per_batch {
            //             let a_val = if batch_self == 1 {
            //                 temp_data[i]
            //             } else {
            //                 temp_data[batch * elements_per_batch + i]
            //             };
            //             let b_val = if batch_right == 1 {
            //                 temp_right_data[i]
            //             } else {
            //                 temp_right_data[batch * elements_per_batch + i]
            //             };
            //             chunk[i] = a_val + b_val;
            //         }
            //     });
            // new_data
        }
    }
    /// Функция активации ReLu
    pub fn relu(&self) -> Vec<f32> {
        self.data
            .par_iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect()
    }

    pub fn pad(&self, padding: (usize, usize, usize, usize)) -> Self {
        assert_eq!(
            self.shape.len(),
            4,
            "Размерность обязательно должна быть четвёртой"
        );
        let (pad_left, pad_right, pad_top, pad_bot) = padding;
        let (n_images, n_layers, row_original, column_original) =
            (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (row_output, column_output) = (
            row_original + pad_top + pad_bot,
            column_original + pad_left + pad_right,
        );
        let require_grad = self.require_grad;

        let data = self.data.clone();
        let mut output: Vec<f32> = vec![0.0; n_images * n_layers * row_output * column_output];

        data.chunks(column_original * row_original)
            .enumerate()
            .for_each(|(layer_index, layer)| {
                layer
                    .chunks(column_original)
                    .enumerate()
                    .for_each(|(row_index, row)| {
                        let offset = (column_output * row_output * layer_index)  // сдвиг по слоям
                            + (pad_top * column_output + pad_left)              // смещение внутри слоя
                            + (row_index * column_output); // сдвиг по строкам
                        for (index, value) in row.iter().enumerate() {
                            output[offset + index] = *value;
                        }
                    });
            });
        TensorData::new(
            output,
            vec![n_images, n_layers, row_output, column_output],
            require_grad,
            vec![],
            Some(Operation::Padding(
                padding.0, padding.1, padding.2, padding.3,
            )),
        )
    }

    /// Возвращает транспонировнное значение тензора
    pub fn transpose(&self) -> TensorData {
        assert_eq!(
            self.shape.len(),
            2,
            "Только матрицы поддаются транспонированию. Дан {}-мерный тензор.",
            self.shape.len()
        );
        let m = self.shape[0];
        let n = self.shape[1];
        let data = &self.data;

        let mut new_data = vec![0.0; m * n];

        new_data.par_chunks_mut(m).enumerate().for_each(|(j, row)| {
            for i in 0..m {
                row[i] = data[i * n + j];
            }
        });
        TensorData::new(new_data, vec![n, m], self.require_grad, vec![], None)
    }

    pub fn sigmoid(&self) -> Vec<f32> {
        let output: Vec<f32> = self
            .data
            .par_iter()
            .map(|&value| 1. / (1. + (-value).exp()))
            .collect();
        output
    }

    pub fn tanh(&self) -> Vec<f32> {
        let output: Vec<f32> = self.data.par_iter().map(|&value| value.tanh()).collect();
        output
    }

    pub fn softmax(&self) -> Vec<f32> {
        assert_eq!(
            self.shape.len(),
            2,
            "softmax работает только для двумерных матриц. Дан {}-мерный тензор.",
            self.shape.len()
        );
        let cols = self.shape[1];
        let data = &self.data;
        let mut output = vec![0.0; self.data.len()];

        output
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(index, row_out)| {
                let start = index * cols;

                // Находим максимум в строке, что бы избежать переполнения на высоких числах
                let mut max_val = data[start];
                for j in 1..cols {
                    let val = data[start + j];
                    if val > max_val {
                        max_val = val;
                    }
                }

                let mut sum = 0.0;
                for j in 0..cols {
                    let val = (data[start + j] - max_val).exp();
                    row_out[j] = val;
                    sum += val;
                }

                for j in 0..cols {
                    row_out[j] /= sum;
                }
            });

        output
    }

    pub fn cross_entropy_with_softmax(&self, targets: &TensorData) -> (Vec<f32>, f32) {
        assert_eq!(
            self.shape.len(),
            2,
            "Кросс-энтропия работает только для двумерных матриц. Дан недвумерный тензор"
        );
        assert_eq!(
            self.shape, targets.shape,
            "Формы предсказаний и целей должны совпадать"
        );

        let rows = self.shape[0];
        let cols = self.shape[1];

        let softmax_output = self.softmax();

        let target_data = targets.data.clone();
        let total_loss: f32 = (0..rows)
            .into_par_iter()
            .map(|r| {
                let start = r * cols;
                let mut row_loss = 0.0;
                for j in 0..cols {
                    let prob = softmax_output[start + j];
                    let target = target_data[start + j];
                    if target > 0.0 {
                        let eps = 1e-7;
                        row_loss += -target * (prob + eps).ln();
                    }
                }
                row_loss
            })
            .sum();

        let loss_value = total_loss / rows as f32;

        (softmax_output, loss_value)
    }
    /// Возвращает объект, `data` в котором равна оригинальному `grad`
    pub fn grad(&self) -> TensorData {
        TensorData::new(self.grad.clone(), self.shape.clone(), false, vec![], None)
    }

    pub fn mse(&self, right_tensor: &TensorData) -> f32 {
        assert_eq!(
            self.shape, right_tensor.shape,
            "Предсказания и таргеты имеют разные размерности"
        );

        let n = self.data.len() as f32;

        // Вычисляем квадрат разности для каждого элемента
        let squared_diff: Vec<f32> = self
            .data
            .par_iter()
            .zip(right_tensor.data.par_iter())
            .map(|(&pred, &target)| {
                let diff = pred - target;
                diff * diff
            })
            .collect();

        let sum: f32 = squared_diff.par_iter().sum();
        let mse_value = sum / n;
        mse_value
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> TensorData {
        let output = TensorData::new(
            self.data.clone(),
            new_shape,
            self.require_grad,
            vec![],
            Some(Operation::Reshape),
        );
        output
    }

    pub fn flatten(&self) -> TensorData {
        let output = self.reshape(vec![
            self.shape[0],
            self.shape[1] * self.shape[2] * self.shape[3],
        ]);
        output
    }

    pub fn max_pool(&self, kernel_shape: (usize, usize), stride: (usize, usize)) -> TensorData {
        let shape = self.shape.clone();
        let layer_data = self.data.clone();
        let x_iter = (shape[shape.len() - 2] - kernel_shape.0) / stride.0 + 1;
        let y_iter = (shape[shape.len() - 1] - kernel_shape.1) / stride.1 + 1;

        let mut layer_im_2_col_data: Vec<f32> =
            Vec::with_capacity(x_iter * y_iter * shape[0] * shape[1]);

        for image in 0..shape[0] {
            for layer in 0..shape[1] {
                for x in 0..x_iter {
                    for y in 0..y_iter {
                        for local_x in 0..kernel_shape.0 {
                            for local_y in 0..kernel_shape.1 {
                                let global_index = image * shape[1] * shape[2] * shape[3] // images
                                    + layer * shape[2] * shape[3] // layers
                                    + x * stride.0 * shape[3] + local_x * shape[3] // x
                                    + y * stride.1 + local_y; // y

                                layer_im_2_col_data.push(layer_data[global_index]);
                            }
                        }
                    }
                }
            }
        }
        // println!("im2col: {:?}", layer_im_2_col_data);

        // math
        let temp_data: Vec<f32> = layer_im_2_col_data
            .par_chunks(kernel_shape.0 * kernel_shape.1)
            .map(|row| row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
            .collect();
        // println!("temp_data: {:?}", temp_data);

        let temp_output = TensorData::new(
            temp_data,
            vec![shape[0], shape[1], x_iter, y_iter],
            self.require_grad, // Уточнить
            vec![],
            Some(Operation::MaxPool(kernel_shape, stride)),
        );
        temp_output
    }
}

/// Объект Тензора. Хранит в себе `TensorData`
#[derive(Debug)]
pub struct Tensor {
    pub tensor_data: Rc<RefCell<TensorData>>,
}

impl Tensor {
    /// Возвращает новый тензор
    pub fn new(
        data: Vec<f32>,
        shape: Vec<usize>,
        require_grad: bool,
        parents: Vec<Rc<RefCell<TensorData>>>,
        operation: Option<Operation>,
    ) -> Tensor {
        let size = shape.iter().product();
        assert_eq!(
            data.len(),
            size,
            "Невозможно конвертировать массив длины {} в матрицу размерностью {:?}",
            data.len(),
            shape
        );
        let tensor_data = TensorData::new(data, shape, require_grad, parents, operation);
        Tensor {
            tensor_data: Rc::new(RefCell::new(tensor_data)),
        }
    }
    /// Возвращает новый тензор, заполненый случайными значениями
    #[allow(dead_code)]
    pub fn uniform(low: f32, high: f32, shape: Vec<usize>, require_grad: bool) -> Tensor {
        let len = shape.iter().product();
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..len).map(|_| rng.random_range(low..high)).collect();
        return Self::new(data, shape, require_grad, Vec::new(), None);
    }

    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let img = ImageReader::open(&path)
            .map_err(|e| format!("Не удалось открыть файл: {}", e))?
            .decode()
            .map_err(|e| format!("Не удалось декодировать изображение: {}", e))?;

        let (width, height) = img.dimensions();
        let channels = match img {
            DynamicImage::ImageLuma8(_) => 1,
            DynamicImage::ImageRgb8(_) => 3,
            DynamicImage::ImageRgba8(_) => 4,
            _ => 3, // fallback
        };

        let mut data = Vec::with_capacity((width * height * channels) as usize);

        // Собираем пиксели в формате [channels, height, width]
        match img {
            DynamicImage::ImageLuma8(img) => {
                // Grayscale: один канал
                for pixel in img.pixels() {
                    data.push(pixel[0] as f32 / 255.0);
                }
            }
            DynamicImage::ImageRgb8(img) => {
                // RGB: переставляем в формат [R..., G..., B...]
                let size = (width * height) as usize;
                let mut red = vec![0.0; size];
                let mut green = vec![0.0; size];
                let mut blue = vec![0.0; size];

                for (i, pixel) in img.pixels().enumerate() {
                    red[i] = pixel[0] as f32 / 255.0;
                    green[i] = pixel[1] as f32 / 255.0;
                    blue[i] = pixel[2] as f32 / 255.0;
                }

                data.extend(red);
                data.extend(green);
                data.extend(blue);
            }
            DynamicImage::ImageRgba8(img) => {
                // RGBA: берём RGB, игнорируем альфа
                let size = (width * height) as usize;
                let mut red = vec![0.0; size];
                let mut green = vec![0.0; size];
                let mut blue = vec![0.0; size];

                for (i, pixel) in img.pixels().enumerate() {
                    red[i] = pixel[0] as f32 / 255.0;
                    green[i] = pixel[1] as f32 / 255.0;
                    blue[i] = pixel[2] as f32 / 255.0;
                }

                data.extend(red);
                data.extend(green);
                data.extend(blue);
            }
            _ => {
                return Err("Неподдерживаемый формат изображения".to_string());
            }
        }

        // Форма: [channels, height, width]
        let shape = vec![channels as usize, height as usize, width as usize];

        Ok(Tensor::new(data, shape, false, vec![], None))
    }

    /// Обнуляет градиент, что бы он не накапливался после `backward()`
    #[allow(dead_code)]
    pub fn zero_grad(&self) {
        for g in &mut self.tensor_data.borrow_mut().grad {
            *g = 0.0
        }
    }
    /// Векторное умножение тензоров
    #[allow(dead_code)]
    pub fn matmul(&self, right_tensor: &Tensor) -> Tensor {
        let output_data = self
            .tensor_data
            .borrow()
            .matmul(&right_tensor.tensor_data.borrow());

        let require_grad = {
            if self.tensor_data.borrow().require_grad
                || right_tensor.tensor_data.borrow().require_grad
            {
                true
            } else {
                false
            }
        };

        Tensor::new(
            output_data,
            vec![
                //Пока что жёстко хардкодим количество размерностей
                self.tensor_data.borrow().shape[0],
                right_tensor.tensor_data.borrow().shape[1],
            ],
            require_grad,
            vec![self.tensor_data.clone(), right_tensor.tensor_data.clone()],
            Some(Operation::Matmul),
        )
    }

    /// Функция активации ReLu
    #[allow(dead_code)]
    pub fn relu(&self) -> Tensor {
        let output_data = self.tensor_data.borrow().relu();

        Tensor::new(
            output_data,
            self.tensor_data.borrow().shape.clone(),
            self.tensor_data.borrow().require_grad,
            vec![self.tensor_data.clone()],
            Some(Operation::ReLu),
        )
    }

    #[allow(dead_code)]
    pub fn sigmoid(&self) -> Tensor {
        let output = self.tensor_data.borrow().sigmoid();
        let require_grad = {
            if self.tensor_data.borrow().require_grad {
                true
            } else {
                false
            }
        };
        Tensor::new(
            output,
            self.tensor_data.borrow().shape.clone(),
            self.tensor_data.borrow().require_grad,
            vec![self.tensor_data.clone()],
            Some(Operation::Sigmoid),
        )
    }

    #[allow(dead_code)]
    pub fn tanh(&self) -> Tensor {
        let output = self.tensor_data.borrow().tanh();
        Tensor::new(
            output,
            self.tensor_data.borrow().shape.clone(),
            self.tensor_data.borrow().require_grad,
            vec![self.tensor_data.clone()],
            Some(Operation::Tanh),
        )
    }

    #[allow(dead_code)]
    pub fn cross_entropy_with_softmax(&self, target: &Tensor) -> Tensor {
        let (softmax_output, loss) = self
            .tensor_data
            .borrow()
            .cross_entropy_with_softmax(&target.tensor_data.borrow());
        let parent0 = self.tensor_data.clone();
        parent0.borrow_mut().data = softmax_output;

        let require_grad = {
            if self.tensor_data.borrow().require_grad {
                true
            } else {
                false
            }
        };

        Tensor::new(
            vec![loss],
            vec![1, 1],
            require_grad,
            vec![parent0, target.tensor_data.clone()],
            Some(Operation::CrossEntropyWithSoftmax),
        )
    }
    /// Возвращает транспонированное значение тензора
    #[allow(dead_code)]
    pub fn transpose(&self) -> Tensor {
        let m = self.tensor_data.borrow().shape[0];
        let n = self.tensor_data.borrow().shape[1];
        let new_data = self.tensor_data.borrow().transpose();

        Tensor::new(new_data.data, vec![n, m], false, vec![], None)
    }
    /// Суммирует данные двух тензоров. Поддерживает две размерности: такую же, как у родителя и построчное добавление `(shape = vec![1, 1, m])`
    #[allow(dead_code)]
    pub fn add(&self, right_tensor: &Tensor) -> Tensor {
        let new_data = self
            .tensor_data
            .borrow()
            .add(&right_tensor.tensor_data.borrow());

        let require_grad = {
            if self.tensor_data.borrow().require_grad {
                true
            } else {
                false
            }
        };

        Tensor::new(
            new_data,
            self.tensor_data.borrow().shape.clone(),
            require_grad,
            vec![self.tensor_data.clone(), right_tensor.tensor_data.clone()],
            Some(Operation::Add),
        )
    }
    /// Средняя квадратичная ошибка
    #[allow(dead_code)]
    pub fn mse(&self, target: &Tensor) -> Tensor {
        let output = self.tensor_data.borrow().mse(&target.tensor_data.borrow());

        let require_grad = {
            if self.tensor_data.borrow().require_grad {
                true
            } else {
                false
            }
        };

        Tensor::new(
            vec![output],
            vec![1],
            require_grad,
            vec![self.tensor_data.clone(), target.tensor_data.clone()],
            Some(Operation::MSE),
        )
    }

    pub fn _softmax(&self) -> Tensor {
        let output = self.tensor_data.borrow().softmax();
        Tensor::new(
            output,
            self.tensor_data.borrow().shape.clone(),
            false,
            vec![],
            None,
        )
    }

    pub fn pad(&self, padding: (usize, usize, usize, usize)) -> Tensor {
        let mut output = self.tensor_data.borrow().pad(padding);
        output.parents.push(self.tensor_data.clone());
        Tensor {
            tensor_data: Rc::new(RefCell::new(output)),
        }
    }

    pub fn grad(&self) -> Tensor {
        let grad_tensor_data = self.tensor_data.borrow().grad();
        Tensor {
            tensor_data: Rc::new(RefCell::new(grad_tensor_data)),
        }
    }

    pub fn conv2d(&self, kernel: &Tensor, stride: (usize, usize)) -> Tensor {
        assert!(
            self.tensor_data.borrow().shape.len() == 4,
            "Свёртку можно проводить только над тензором размерности 4. Получено: {}",
            self.tensor_data.borrow().shape.len()
        );
        assert!(
            kernel.tensor_data.borrow().shape.len() == 4,
            "Свёртку можно проводить только ядром размерности 4. Получено: {}",
            kernel.tensor_data.borrow().shape.len()
        );
        let layer_shape = self.tensor_data.borrow().shape.clone();
        let kernel_shape = kernel.tensor_data.borrow().shape.clone();
        let kernel_data = kernel.tensor_data.borrow().data.clone();

        let x_iter = (layer_shape[layer_shape.len() - 2] - kernel_shape[kernel_shape.len() - 2])
            / stride.0
            + 1;
        let y_iter = (layer_shape[layer_shape.len() - 1] - kernel_shape[kernel_shape.len() - 1])
            / stride.1
            + 1;

        // ===== im2col =====
        let temp_tensor = im2col(self.clone(), kernel_shape.clone(), stride.clone());
        temp_tensor.tensor_data.borrow_mut().require_grad = true;

        // ===== Подготовка ядра =====
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

        // ===== matmul =====
        let output = temp_tensor.matmul(&temp_kernel);
        let output_clone = temp_tensor.matmul(&temp_kernel);

        // ===== col2im =====
        output.tensor_data.borrow_mut().operation = Some(Operation::Conv2d(stride, output_clone));
        output.tensor_data.borrow_mut().shape =
            vec![layer_shape[0], kernel_shape[0], x_iter, y_iter];

        let output_data = output.tensor_data.borrow().data.clone();
        let mut temp_output_data = vec![0.0; output_data.len()];

        let layer_len = x_iter * y_iter;
        let image_len = kernel_shape[0] * layer_len;

        temp_output_data
            .par_chunks_mut(kernel_shape[0] * x_iter * y_iter)
            .enumerate()
            .for_each(|(image, chunk)| {
                for y in 0..y_iter {
                    for layer in 0..kernel_shape[0] {
                        for x in 0..x_iter {
                            let local_index = layer * layer_len + x * y_iter + y;
                            let global_index = local_index + image * image_len;
                            chunk[local_index] = output_data[global_index];
                        }
                    }
                }
            });

        output.tensor_data.borrow_mut().data = temp_output_data;
        output
            .tensor_data
            .borrow_mut()
            .parents
            .push(self.tensor_data.clone());
        output
            .tensor_data
            .borrow_mut()
            .parents
            .push(kernel.tensor_data.clone());

        output
    }

    pub fn shape(&self) -> Vec<usize> {
        self.tensor_data.borrow().shape.clone()
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let mut output_tensor_data = self.tensor_data.borrow().reshape(new_shape);
        output_tensor_data.parents.push(self.tensor_data.clone());
        let output = Tensor {
            tensor_data: Rc::new(RefCell::new(output_tensor_data)),
        };
        output
    }

    pub fn flatten(&self) -> Tensor {
        let mut output_tensor_data = self.tensor_data.borrow().flatten();
        output_tensor_data.parents.push(self.tensor_data.clone());
        let output = Tensor {
            tensor_data: Rc::new(RefCell::new(output_tensor_data)),
        };
        output
    }

    pub fn max_pool(&self, kernel_shape: (usize, usize), stride: (usize, usize)) -> Tensor {
        let mut output = self.tensor_data.borrow().max_pool(kernel_shape, stride);
        output.parents.push(self.tensor_data.clone());
        Tensor {
            tensor_data: Rc::new(RefCell::new(output)),
        }
    }

    fn add_grad(parent: Rc<RefCell<TensorData>>, new_grad: Vec<f32>) {
        // println!("add_grad new_grad: {:?}", new_grad);
        let mut pipa = parent.borrow_mut();
        if pipa.grad.is_empty() {
            pipa.grad = new_grad;
        } else {
            for (existing, &new) in pipa.grad.iter_mut().zip(new_grad.iter()) {
                *existing += new;
            }
        }
    }
    /// Функция взятия производной. Строит граф вычислений, берёт производные по каждому изначальному тензору.
    pub fn backward(&self) {
        let start_node = self.tensor_data.clone();
        let basic_grad = vec![1.; start_node.borrow().shape.iter().product()];
        if start_node.borrow().grad.len() == 0 {
            start_node.borrow_mut().grad = basic_grad;
        }
        let graph = graph::Graph::new(start_node).build_topo();
        // println!("graph size: {}", graph.len());

        for i in graph.iter().rev() {
            if !i.borrow().require_grad || i.borrow().operation.is_none() {
                continue;
            }

            // println!("Operation: {:?}", i.borrow().operation);
            // println!("grad: {:?}", i.borrow().grad);

            let current_node = i.borrow();
            let current_node_grad = TensorData::new(
                current_node.grad.clone(),
                current_node.shape.clone(),
                false,
                vec![],
                None,
            );

            // let pipa = op_name(&current_node.operation);
            // addictive_functions::print_grad_stats(pipa, &current_node_grad.data);

            match current_node.operation {
                Some(Operation::Matmul) => {
                    let grad_left =
                        current_node_grad.matmul(&current_node.parents[1].borrow().transpose());
                    let grad_right = current_node.parents[0]
                        .borrow()
                        .transpose()
                        .matmul(&current_node_grad);

                    Self::add_grad(current_node.parents[0].clone(), grad_left);
                    Self::add_grad(current_node.parents[1].clone(), grad_right);
                }
                Some(Operation::Add) => {
                    let right_shape = current_node.parents[1].borrow().shape.clone();

                    // Градиент для левого (большого) — без изменений
                    Self::add_grad(
                        current_node.parents[0].clone(),
                        current_node_grad.data.clone(),
                    );

                    // Градиент для правого (маленького) — суммируем по батчам
                    let elements_per_batch: usize = right_shape.iter().product();
                    let mut grad_right = vec![0.0; elements_per_batch];

                    for chunk in current_node_grad.data.chunks(elements_per_batch) {
                        for (i, &g) in chunk.iter().enumerate() {
                            grad_right[i] += g;
                        }
                    }

                    Self::add_grad(current_node.parents[1].clone(), grad_right);
                }
                Some(Operation::ReLu) => {
                    let grad = current_node
                        .data
                        .par_iter()
                        .zip(current_node_grad.data.par_iter())
                        .map(|(&data, &grad)| if data > 0.0 { grad } else { 0.0 })
                        .collect();
                    Self::add_grad(current_node.parents[0].clone(), grad);
                }
                Some(Operation::Sigmoid) => {
                    let grad_input: Vec<f32> = current_node
                        .data
                        .par_iter()
                        .zip(current_node_grad.data.par_iter())
                        .map(|(&data, &grad)| grad * data * (1.0 - data))
                        .collect();
                    Self::add_grad(current_node.parents[0].clone(), grad_input);
                }
                Some(Operation::Tanh) => {
                    let grad_input: Vec<f32> = current_node
                        .data
                        .par_iter()
                        .zip(current_node_grad.data.par_iter())
                        .map(|(&data, &grad)| (1.0 - data * data) * grad)
                        .collect();
                    Self::add_grad(current_node.parents[0].clone(), grad_input);
                }
                Some(Operation::MSE) => {
                    let grad: Vec<f32> = {
                        let y_pred = current_node.parents[0].borrow();
                        let y_true = current_node.parents[1].borrow();
                        let n = y_pred.data.len() as f32;
                        let grad_output = &current_node_grad.data[0];

                        y_pred
                            .data
                            .par_iter()
                            .zip(y_true.data.par_iter())
                            .map(|(&pred, &true_val)| 2.0 * (pred - true_val) * grad_output / n)
                            .collect()
                    };
                    Self::add_grad(current_node.parents[0].clone(), grad);
                }
                Some(Operation::CrossEntropyWithSoftmax) => {
                    let grad: Vec<f32> = {
                        let y_pred = current_node.parents[0].borrow();
                        let y_true = current_node.parents[1].borrow();
                        let n = y_pred.shape[0] as f32;
                        y_pred
                            .data
                            .iter()
                            .zip(y_true.data.iter())
                            .map(|(&pred, &true_val)| (pred - true_val) / n)
                            .collect()
                    };

                    Self::add_grad(current_node.parents[0].clone(), grad);
                }
                Some(Operation::Conv2d((stride_x, stride_y), ref matmul_after_im2col)) => {
                    // 1. Достаём родителей ДО backward (нужны нам для форм)
                    let (input_col, kernel) = {
                        let td = matmul_after_im2col.tensor_data.borrow();
                        (td.parents[0].clone(), td.parents[1].clone())
                    };

                    // 2. Затравка grad'а у matmul_after_im2col
                    //    ВАЖНО: форма должна совпадать с формой matmul_after_im2col в прямом проходе!
                    //    Если matmul_after_im2col имеет форму [N*out_H*out_W, out_C],
                    //    то и grad должен быть той же формы.
                    let grad_shape = matmul_after_im2col.tensor_data.borrow().shape.clone();
                    matmul_after_im2col.tensor_data.borrow_mut().grad =
                        current_node_grad.data.clone();
                    // если нужно reshape — сделай его здесь:
                    // matmul_after_im2col.tensor_data.borrow_mut().grad =
                    //     Some(reshape(current_node_grad.data.clone(), grad_shape));

                    // 3. Один backward прогоняет градиент через matmul → в im2col(x) и в W
                    matmul_after_im2col.backward();

                    // 4. Теперь grad_col и grad_W уже посчитаны
                    let grad_col = input_col.borrow().grad.clone();
                    let grad_w = kernel.borrow().grad.clone();

                    // 5. col2im раскидывает градиент по im2col обратно в форму входа
                    let input_shape = current_node.parents[2].borrow().shape.clone(); // [N, C, H, W]
                    let kernel_shape = current_node.parents[3].borrow().shape.clone(); // [out_C, C, kH, kW]
                    let n = input_shape[0];
                    let c = input_shape[1];
                    let h = input_shape[2];
                    let w = input_shape[3];
                    let kh = kernel_shape[2];
                    let kw = kernel_shape[3];
                    let out_h = (h - kh) / stride_x + 1;
                    let out_w = (w - kw) / stride_y + 1;

                    let grad_col_tensor = Tensor::new(
                        grad_col,
                        vec![n * out_h * out_w, c * kh * kw],
                        false,
                        vec![],
                        None,
                    );

                    let layer_grad = col2im(
                        grad_col_tensor,
                        input_shape,
                        kernel_shape,
                        (stride_x, stride_y),
                    )
                    .tensor_data
                    .borrow()
                    .data
                    .clone();

                    // 6. Градиент по W — уже правильной формы (reshape не нужен, если shape совпадает)
                    //    Если grand_parents1.grad имеет форму [out_C, C*kH*kW], а тебе нужна
                    //    форма [out_C, C, kH, kW] — reshape'ни:
                    let kernel_grad = grad_w; // при необходимости: reshape(grad_w, kernel_shape)

                    Self::add_grad(current_node.parents[2].clone(), layer_grad);
                    Self::add_grad(current_node.parents[3].clone(), kernel_grad);
                }
                Some(Operation::Padding(pad_top, pad_bot, pad_left, pad_right)) => {
                    let grad_output = &current_node_grad.data;
                    let grad_output_shape = &current_node_grad.shape;

                    // Оригинальная форма входа (до паддинга)
                    let n = grad_output_shape[0];
                    let c = grad_output_shape[1];
                    let h_orig = grad_output_shape[2] - pad_left - pad_right;
                    let w_orig = grad_output_shape[3] - pad_top - pad_bot;

                    let mut grad_input = vec![0.0; n * c * h_orig * w_orig];

                    // Копируем только центральную часть (без паддинга)
                    for batch in 0..n {
                        for layer in 0..c {
                            for y in 0..h_orig {
                                for x in 0..w_orig {
                                    let src_idx =
                                        batch * c * grad_output_shape[2] * grad_output_shape[3]
                                            + layer * grad_output_shape[2] * grad_output_shape[3]
                                            + (y + pad_top) * grad_output_shape[3]
                                            + (x + pad_left);

                                    let dst_idx = batch * c * h_orig * w_orig
                                        + layer * h_orig * w_orig
                                        + y * w_orig
                                        + x;

                                    grad_input[dst_idx] = grad_output[src_idx];
                                }
                            }
                        }
                    }

                    Self::add_grad(current_node.parents[0].clone(), grad_input);
                }
                Some(Operation::Reshape) => {
                    // let grad = vec![1.0; current_node_grad.shape.clone().iter().product()];
                    let grad = current_node_grad.data.clone();
                    Self::add_grad(current_node.parents[0].clone(), grad);
                }
                Some(Operation::MaxPool(kernel_shape, stride)) => {
                    let current_shape = current_node.shape.clone();
                    let parent_shape = current_node.parents[0].borrow().shape.clone();
                    let mut grad: Vec<f32> =
                        vec![0.0; current_node.parents[0].borrow().data.capacity()];
                    let parent_data = current_node.parents[0].borrow().data.clone();
                    for image in 0..parent_shape[0] {
                        for layer in 0..parent_shape[1] {
                            for x in 0..current_shape[2] {
                                for y in 0..current_shape[3] {
                                    let current_index: usize = image * current_shape[1] * current_shape[2] * current_shape[3] // images
                                                + layer * current_shape[2] * current_shape[3] // layers
                                                + x * current_shape[3] // x
                                                + y; // y
                                    let mut max_index = 0;
                                    let mut max_value = f32::NEG_INFINITY;
                                    for local_x in 0..kernel_shape.0 {
                                        for local_y in 0..kernel_shape.1 {
                                            let parent_index = image * parent_shape[1] * parent_shape[2] * parent_shape[3] // images
                                                + layer * parent_shape[2] * parent_shape[3] // layers
                                                + x * stride.0 * parent_shape[3] + local_x * parent_shape[3] // x
                                                + y * stride.1 + local_y; // y
                                            if parent_data[parent_index] > max_value {
                                                max_index = parent_index;
                                                max_value = parent_data[parent_index]
                                            }
                                        }
                                    }
                                    grad[max_index] += current_node_grad.data[current_index];
                                }
                            }
                        }
                    }

                    Self::add_grad(current_node.parents[0].clone(), grad);
                }
                None => {}
            }
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.tensor_data.borrow();
        let shape = &data.shape;

        if data.data.is_empty() {
            return write!(f, "Tensor {:?}: []", shape);
        }

        write!(f, "Tensor {:?}: ", shape)?;

        // Внутренняя рекурсивная функция
        fn fmt_recursive(
            f: &mut fmt::Formatter<'_>,
            data: &[f32],
            shape: &[usize],
            offset: usize,
            depth: usize,
        ) -> fmt::Result {
            if shape.is_empty() {
                return write!(f, "{:.4}", data[offset]);
            }

            if shape.len() == 1 {
                write!(f, "[")?;
                for i in 0..shape[0] {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:.4}", data[offset + i])?;
                }
                return write!(f, "]");
            }

            let dim = shape[0];
            let rest = &shape[1..];
            let step: usize = rest.iter().product();

            write!(f, "[\n")?;

            for i in 0..dim {
                for _ in 0..depth + 1 {
                    write!(f, " ")?;
                }
                fmt_recursive(f, data, rest, offset + i * step, depth + 1)?;
                if i < dim - 1 {
                    write!(f, ",")?;
                }
                writeln!(f)?;
            }

            for _ in 0..depth {
                write!(f, " ")?;
            }
            write!(f, "]")
        }

        fmt_recursive(f, &data.data, shape, 0, 0)
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            tensor_data: self.tensor_data.clone(),
        }
    }
}
