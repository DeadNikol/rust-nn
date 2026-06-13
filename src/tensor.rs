//! В этом крейте прописаны классы Тензора, данных тензора, функции над тензорами, операция взятия производной, перечисление операций

use crate::graph;
use rand::RngExt;
use rayon::prelude::*;
use std::{
    cell::RefCell,
    fmt::{self},
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
}

/// Внутренние данные тензора. Хранятся как Rc<RefCell<TensorData>>>
pub struct TensorData {
    /// Сами данные тензора
    pub data: Vec<f32>, // Само значение тензора. Оно будет плоским, обращение к строкам и тому подобное будет только через индексы. Срезов, как numpy пока не планируется
    /// Размерность тензора. Пока что она обязательно должна быть трёхмерной
    pub shape: Vec<usize>, // Размерность тензора. Она должна быть строго трёхмерной
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
    pub fn zero_grad(&mut self) {
        for g in &mut self.grad {
            *g = 0.0
        }
    }
    /// Векторное умножение данных двух тензоров
    pub fn matmul(&self, right_tensor: &TensorData) -> Vec<f32> {
        let batch_shape_flag = self.shape[0] == 1 && right_tensor.shape[0] == 1; // Если это не плоские матрицы
        assert_eq!(
            batch_shape_flag, true,
            "Матрично можно перемножать только матрицы (shape[0] == 1). Дан тензор"
        );

        let shape_flag: bool = // флаг совпадения размерностей
                self.shape[2] == right_tensor.shape[1];
        assert_eq!(
            // Если размерности не сошлись по принципу [x,n] @ [n,y] => [x,y], то кидаем ошибку
            shape_flag,
            true,
            "Что бы векторно перемножить две матрицы, их размерности должны быть [x,n] @ [n,y] => [x,y]. Даны [{},{}] @ [{},{}] => ошибка",
            self.shape[1],
            self.shape[2],
            right_tensor.shape[1],
            right_tensor.shape[2]
        );

        let m = self.shape[1];
        let k = self.shape[2];
        let n = right_tensor.shape[2];

        let a = &self.data;
        let b = &right_tensor.data;

        // Параллельная обработка строк
        let output_data: Vec<f32> = (0..m)
            .into_par_iter()
            .flat_map(|i| {
                let mut row = vec![0.0; n];
                for l in 0..k {
                    let a_il = a[i * k + l];
                    let b_row = &b[l * n..(l + 1) * n];
                    for j in 0..n {
                        row[j] += a_il * b_row[j];
                    }
                }
                row
            })
            .collect();
        output_data
    }
    /// Суммирует данные двух тензоров. Поддерживает две размерности: такую же, как у родителя и построчное добавление (shape = vec![1, 1, m])
    pub fn add(&self, right_tensor: &TensorData) -> Vec<f32> {
        assert_eq!(self.shape[0], 1);
        assert_eq!(right_tensor.shape[0], 1);
        assert_eq!(
            self.shape[2], right_tensor.shape[2],
            "Число столбцов должно совпадать"
        );

        let rows_self = self.shape[1];
        let rows_right = right_tensor.shape[1];
        let cols = self.shape[2];
        let a = &self.data;
        let b = &right_tensor.data;

        if rows_self == rows_right {
            // Одинаковая высота – поэлементное сложение (возможно, одинаковые формы)
            assert_eq!(self.shape, right_tensor.shape);
            let mut new_data = vec![0.0; a.len()];
            new_data.par_iter_mut().enumerate().for_each(|(i, v)| {
                *v = a[i] + b[i];
            });
            new_data
        } else if rows_self == 1 && rows_right > 1 {
            // self – строка (bias), right – матрица
            let bias = a;
            let mut new_data = vec![0.0; b.len()];
            new_data
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row_idx, row)| {
                    let base = row_idx * cols;
                    for c in 0..cols {
                        row[c] = bias[c] + b[base + c];
                    }
                });
            new_data
        } else if rows_right == 1 && rows_self > 1 {
            // right – строка (bias), self – матрица
            let bias = b;
            let mut new_data = vec![0.0; a.len()];
            new_data
                .par_chunks_mut(cols)
                .enumerate()
                .for_each(|(row_idx, row)| {
                    let base = row_idx * cols;
                    for c in 0..cols {
                        row[c] = a[base + c] + bias[c];
                    }
                });
            new_data
        } else {
            panic!(
                "Неподдерживаемые формы для сложения: self.shape={:?}, right.shape={:?}",
                self.shape, right_tensor.shape
            );
        }
    }
    /// Функция активации ReLu
    pub fn relu(&self) -> Vec<f32> {
        self.data
            .par_iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect()
    }
    /// Возвращает транспонировнное значение тензора
    pub fn transpose(&self) -> TensorData {
        let m = self.shape[1];
        let n = self.shape[2];
        let data = &self.data;

        let mut new_data = vec![0.0; m * n];

        new_data.par_chunks_mut(m).enumerate().for_each(|(j, row)| {
            for i in 0..m {
                row[i] = data[i * n + j];
            }
        });

        TensorData {
            data: new_data,
            shape: vec![self.shape[0], n, m],
            require_grad: self.require_grad,
            parents: vec![],
            grad: vec![],
            operation: None,
        }
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
        let cols = self.shape[2];
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
            self.shape, targets.shape,
            "Формы предсказаний и целей должны совпадать"
        );

        let rows = self.shape[1];
        let cols = self.shape[2];

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
        TensorData {
            data: self.grad.clone(),
            shape: self.shape.clone(),
            require_grad: false, // Возможно здесь не стоит хардкодить
            parents: vec![],
            grad: vec![],
            operation: None,
        }
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
}

/// Объект Тензора. Хранит в себе `TensorData`
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
        assert_eq!(
            shape.len(),
            3,
            "Тензор обязан иметь 3 размерности, получено: {}",
            shape.len()
        );

        let size = shape.iter().product();
        assert_eq!(
            data.len(),
            size,
            "Невозможно конвертировать массив длины {} в матрицу размерностью {:?}",
            data.len(),
            shape
        );
        let tensor_data = TensorData {
            data,
            shape,
            require_grad,
            parents,
            grad: Vec::new(),
            operation,
        };
        Tensor {
            tensor_data: Rc::new(RefCell::new(tensor_data)),
        }
    }
    /// Возвращает новый тензор, заполненый случайными значениями
    pub fn uniform(low: f32, high: f32, shape: Vec<usize>, require_grad: bool) -> Tensor {
        let len = shape.iter().product();
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..len).map(|_| rng.random_range(low..high)).collect();
        return Self::new(data, shape, require_grad, Vec::new(), None);
    }

    /// Позволяет получить значение по определённому индексу
    pub fn get(&self, batch: usize, row: usize, column: usize) -> f32 {
        //Раньше функция возвращала ссылку на число

        // Блок ниже можно удалять к хуям

        let selff = self.tensor_data.borrow();
        let batch_len = selff.data.len() / selff.shape[0]; // Количество элементов в одном батче
        let row_len = batch_len / selff.shape[1]; // Количество строк внутри батча
        let lambda_b = batch * batch_len; // Сдвиг по батчу
        let lambda_r = row * row_len; // Свдиг по строке
        let index = lambda_b + lambda_r + column;
        let output = selff.data.get(index).cloned();

        // let batch_len = self.data.len() / self.shape[0]; // Количество элементов в одном батче
        // let row_len = batch_len / self.shape[1]; // Количество строк внутри батча
        // let lambda_b = batch * batch_len; // Сдвиг по батчу
        // let lambda_r = row * row_len; // Свдиг по строке
        // let index = lambda_b + lambda_r + column;
        // let output = self.data.get(index);
        match output {
            Some(value) => return value,
            None => {
                panic!(
                    "Не сущиествует элемента с индексом {}, {}, {}. Номер элемента: {}",
                    batch, row, column, index
                );
            }
        }
    }
    /// Обнуляет градиент, что бы он не накапливался после `backward()`
    #[allow(dead_code)]
    pub fn zero_grad(&self) {
        for g in &mut self.tensor_data.borrow_mut().grad {
            *g = 0.0
        }
    }
    /// Векторное умножение тензоров
    pub fn matmul(&self, right_tensor: &Tensor) -> Tensor {
        let output_data = self
            .tensor_data
            .borrow()
            .matmul(&right_tensor.tensor_data.borrow());

        Tensor::new(
            output_data,
            vec![
                1,
                self.tensor_data.borrow().shape[1],
                right_tensor.tensor_data.borrow().shape[2],
            ],
            false,
            vec![self.tensor_data.clone(), right_tensor.tensor_data.clone()],
            Some(Operation::Matmul),
        )
    }
    /// Функция активации ReLu
    pub fn relu(&self) -> Tensor {
        let output_data = self.tensor_data.borrow().relu();
        Tensor::new(
            output_data,
            self.tensor_data.borrow().shape.clone(),
            false,
            vec![self.tensor_data.clone()],
            Some(Operation::ReLu),
        )
    }

    pub fn sigmoid(&self) -> Tensor {
        let output = self.tensor_data.borrow().sigmoid();
        Tensor::new(
            output,
            self.tensor_data.borrow().shape.clone(),
            true,
            vec![self.tensor_data.clone()],
            Some(Operation::Sigmoid),
        )
    }

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

    pub fn cross_entropy_with_softmax(&self, target: &Tensor) -> Tensor {
        let (softmax_output, loss) = self
            .tensor_data
            .borrow()
            .cross_entropy_with_softmax(&target.tensor_data.borrow());
        let parent0 = self.tensor_data.clone();
        parent0.borrow_mut().data = softmax_output;
        Tensor::new(
            vec![loss],
            vec![1, 1, 1],
            true,
            vec![parent0, target.tensor_data.clone()],
            Some(Operation::CrossEntropyWithSoftmax),
        )
    }
    /// Возвращает транспонированное значение тензора
    #[allow(dead_code)]
    pub fn transpose(&self) -> Tensor {
        let m = self.tensor_data.borrow().shape[1];
        let n = self.tensor_data.borrow().shape[2];
        let new_data = self.tensor_data.borrow().transpose();

        Tensor::new(new_data.data, vec![1, n, m], false, vec![], None)
    }
    /// Суммирует данные двух тензоров. Поддерживает две размерности: такую же, как у родителя и построчное добавление (shape = vec![1, 1, m])
    #[allow(dead_code)]
    pub fn add(&self, right_tensor: &Tensor) -> Tensor {
        let m = std::cmp::max(
            self.tensor_data.borrow().shape[1],
            right_tensor.tensor_data.borrow().shape[1],
        );
        let n = self.tensor_data.borrow().shape[2];

        let new_data = self
            .tensor_data
            .borrow()
            .add(&right_tensor.tensor_data.borrow());

        Tensor::new(
            new_data,
            vec![1, m, n],
            false,
            vec![self.tensor_data.clone(), right_tensor.tensor_data.clone()],
            Some(Operation::Add),
        )
    }
    /// Средняя квадратичная ошибка
    pub fn mse(&self, y_true: &Tensor) -> Tensor {
        let output = self.tensor_data.borrow().mse(&y_true.tensor_data.borrow());
        Tensor::new(
            vec![output],
            vec![1, 1, 1],
            false,
            vec![self.tensor_data.clone(), y_true.tensor_data.clone()],
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

    pub fn grad(&self) -> Tensor {
        let grad_tensor_data = self.tensor_data.borrow().grad();
        Tensor {
            tensor_data: Rc::new(RefCell::new(grad_tensor_data)),
        }
    }
    /// Функция взятия производной. Строит граф вычислений, берёт производные по каждому изначальному тензору.
    fn add_grad(parent: Rc<RefCell<TensorData>>, new_grad: Vec<f32>) {
        let mut pipa = parent.borrow_mut();
        if pipa.grad.is_empty() {
            pipa.grad = new_grad;
        } else {
            for (existing, &new) in pipa.grad.iter_mut().zip(new_grad.iter()) {
                *existing += new;
            }
        }
    }
    pub fn backward(&self) {
        let start_node = self.tensor_data.clone();
        let basic_grad = vec![1.; start_node.borrow().shape.iter().product()];
        start_node.borrow_mut().grad = basic_grad;
        let graph = graph::Graph::new(start_node).build_topo();

        for i in graph.iter().rev() {
            let current_node = i.borrow();
            let current_node_grad = TensorData {
                data: current_node.grad.clone(),
                shape: current_node.shape.clone(),
                require_grad: false,
                parents: vec![],
                grad: vec![],
                operation: None,
            };

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
                    let grad = current_node_grad.data.clone();
                    let rows_self = current_node.parents[0].borrow().shape[1];
                    let rows_right = current_node.parents[1].borrow().shape[1];
                    let cols = current_node.shape[2];

                    if rows_self == 1 && rows_right > 1 {
                        let mut summed_grad = vec![0.0; cols];
                        for chunk in grad.chunks(cols) {
                            for (s, &g) in summed_grad.iter_mut().zip(chunk.iter()) {
                                *s += g;
                            }
                        }
                        Self::add_grad(current_node.parents[0].clone(), summed_grad);
                        Self::add_grad(current_node.parents[1].clone(), grad);
                    } else if rows_right == 1 && rows_self > 1 {
                        let mut summed_grad = vec![0.0; cols];
                        for chunk in grad.chunks(cols) {
                            for (s, &g) in summed_grad.iter_mut().zip(chunk.iter()) {
                                *s += g;
                            }
                        }
                        Self::add_grad(current_node.parents[0].clone(), grad);
                        Self::add_grad(current_node.parents[1].clone(), summed_grad);
                    } else {
                        Self::add_grad(current_node.parents[0].clone(), grad.clone());
                        Self::add_grad(current_node.parents[1].clone(), grad);
                    }
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
                        let n = y_pred.shape[1] as f32;
                        y_pred
                            .data
                            .iter()
                            .zip(y_true.data.iter())
                            .map(|(&pred, &true_val)| (pred - true_val) / n)
                            .collect()
                    };

                    Self::add_grad(current_node.parents[0].clone(), grad);
                }
                None => {}
            }
        }
    }
    // pub fn backward(&self) {
    //     let start_node = self.tensor_data.clone(); // Клонируем данные текущего тензора
    //     let basic_grad = vec![1.; start_node.borrow().shape.iter().product()]; // Создаём градииент по умолчанию
    //     start_node.borrow_mut().grad = basic_grad; // Устанавливаем градиент по умолчанию
    //     let graph = graph::Graph::new(start_node); // Инициализируем граф
    //     let graph = graph.build_topo(); // Собираем граф вычислений

    //     for i in graph.iter().rev() {
    //         // Проходим по графу в обратном порядке
    //         let current_node = i.borrow(); // получаем доступ к данным
    //         let current_node_grad = TensorData {
    //             // Создаём новый объект, где data = grad, потому что для Vec<f32> нет прописанного matmul
    //             data: current_node.grad.clone(),
    //             shape: current_node.shape.clone(),
    //             require_grad: false,
    //             parents: vec![],
    //             grad: vec![],
    //             operation: None,
    //         };
    //         match current_node.operation {
    //             // В зависимости от операции, создавшей этот тензор, вычисляем производную
    //             Some(Operation::Matmul) => {
    //                 current_node.parents[0].borrow_mut().grad =
    //                     current_node_grad.matmul(&current_node.parents[1].borrow().transpose());

    //                 current_node.parents[1].borrow_mut().grad = current_node.parents[0]
    //                     .borrow_mut()
    //                     .transpose()
    //                     .matmul(&current_node_grad);
    //             }
    //             Some(Operation::Add) => {
    //                 let grad = current_node_grad.data.clone();
    //                 let rows_self = current_node.parents[0].borrow().shape[1];
    //                 let rows_right = current_node.parents[1].borrow().shape[1];
    //                 let cols = current_node.shape[2];

    //                 // Левый родитель — bias (1 строка), правый — матрица
    //                 if rows_self == 1 && rows_right > 1 {
    //                     // Суммируем градиент по строкам для bias
    //                     let mut summed_grad = vec![0.0; cols];
    //                     for chunk in grad.chunks(cols) {
    //                         for (s, &g) in summed_grad.iter_mut().zip(chunk.iter()) {
    //                             *s += g;
    //                         }
    //                     }
    //                     current_node.parents[0].borrow_mut().grad = summed_grad;
    //                     current_node.parents[1].borrow_mut().grad = grad;
    //                 }
    //                 // Правый родитель — bias, левый — матрица
    //                 else if rows_right == 1 && rows_self > 1 {
    //                     let mut summed_grad = vec![0.0; cols];
    //                     for chunk in grad.chunks(cols) {
    //                         for (s, &g) in summed_grad.iter_mut().zip(chunk.iter()) {
    //                             *s += g;
    //                         }
    //                     }
    //                     current_node.parents[0].borrow_mut().grad = grad;
    //                     current_node.parents[1].borrow_mut().grad = summed_grad;
    //                 }
    //                 // Обычное сложение одинаковых форм
    //                 else {
    //                     current_node.parents[0].borrow_mut().grad = grad.clone();
    //                     current_node.parents[1].borrow_mut().grad = grad;
    //                 }
    //             }
    //             Some(Operation::ReLu) => {
    //                 current_node.parents[0].borrow_mut().grad = current_node
    //                     .data
    //                     .par_iter()
    //                     .zip(current_node_grad.data.par_iter())
    //                     .map(|(&data, &grad)| if data > 0.0 { grad } else { 0.0 })
    //                     .collect();
    //             }
    //             Some(Operation::Sigmoid) => {
    //                 let sigmoid_output = &current_node.data; // σ(x)
    //                 let grad_output = &current_node_grad.data; // grad от вышестоящих узлов
    //                 // println!("sigmoid: {:?}", sigmoid_output);
    //                 // println!("grad_sigmoid: {:?}", grad_output);
    //                 // Производная сигмоида: σ'(x) = σ(x) * (1 - σ(x))
    //                 // grad_input = grad_output * σ'(x)
    //                 let grad_input: Vec<f32> = sigmoid_output
    //                     .par_iter()
    //                     .zip(grad_output.par_iter())
    //                     .map(|(&data_value, &grad_value)| {
    //                         let derivative = data_value * (1. - data_value);
    //                         let result = grad_value * derivative;
    //                         result
    //                     })
    //                     .collect();

    //                 current_node.parents[0].borrow_mut().grad = grad_input;
    //             }
    //             Some(Operation::Tanh) => {
    //                 let output: Vec<f32> = current_node
    //                     .data
    //                     .par_iter()
    //                     .zip(current_node_grad.data.par_iter())
    //                     .map(|(&data_value, &grad_value)| {
    //                         (1. - data_value * data_value) * grad_value
    //                     })
    //                     .collect();
    //                 current_node.parents[0].borrow_mut().grad = output;
    //             }
    //             Some(Operation::MSE) => {
    //                 let grad = {
    //                     let y_pred = current_node.parents[0].borrow();
    //                     let y_true = current_node.parents[1].borrow();
    //                     let n = y_pred.data.len() as f32;
    //                     let grad_output = &current_node_grad.data[0];

    //                     y_pred
    //                         .data
    //                         .par_iter()
    //                         .zip(y_true.data.par_iter())
    //                         .map(|(&pred_data, &true_data)| {
    //                             2.0 * (pred_data - true_data) * grad_output / n
    //                         })
    //                         .collect::<Vec<f32>>()
    //                 };
    //                 current_node.parents[0].borrow_mut().grad = grad;
    //             }
    //             Some(Operation::CrossEntropyWithSoftmax) => {
    //                 let grad: Vec<f32> = {
    //                     let y_pred = current_node.parents[0].borrow();
    //                     let y_true = current_node.parents[1].borrow();
    //                     let n = y_pred.shape[1] as f32;
    //                     let grad: Vec<f32> = y_pred
    //                         .data
    //                         .iter()
    //                         .zip(y_true.data.iter())
    //                         .map(|(&pred_value, &true_value)| (pred_value - true_value) / n)
    //                         .collect();
    //                     grad
    //                 };
    //                 current_node.parents[0].borrow_mut().grad = grad;
    //             }
    //             None => {
    //                 // println!("Это конечное значение");
    //             }
    //         }
    //     }
    // }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let selff = self.tensor_data.borrow();

        writeln!(
            f,
            "Tensor [{} x {} x {}]:",
            selff.shape[0], selff.shape[1], selff.shape[2]
        )?;

        for b in 0..selff.shape[0] {
            if selff.shape[0] > 1 {
                writeln!(f, "Batch {}:", b)?;
            }
            writeln!(f, "[")?;
            for r in 0..selff.shape[1] {
                write!(f, " [")?;
                for c in 0..selff.shape[2] {
                    let val = self.get(b, r, c);
                    if c > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:.4}", val)?;
                }
                if r == selff.shape[1] - 1 {
                    writeln!(f, "]")?;
                } else {
                    writeln!(f, "],")?;
                }
            }
            writeln!(f, "]")?;
        }
        Ok(())

        // writeln!(
        //     f,
        //     "Tensor [{} x {} x {}]:",
        //     self.shape[0], self.shape[1], self.shape[2]
        // )?;

        // for b in 0..self.shape[0] {
        //     if self.shape[0] > 1 {
        //         writeln!(f, "Batch {}:", b)?;
        //     }
        //     writeln!(f, "[")?;
        //     for r in 0..self.shape[1] {
        //         write!(f, " [")?;
        //         for c in 0..self.shape[2] {
        //             let val = self.get(b, r, c);
        //             if c > 0 {
        //                 write!(f, ", ")?;
        //             }
        //             write!(f, "{:.4}", val)?;
        //         }
        //         if r == self.shape[1] - 1 {
        //             writeln!(f, "]")?;
        //         } else {
        //             writeln!(f, "],")?;
        //         }
        //     }
        //     writeln!(f, "]")?;
        // }
        // Ok(())
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            tensor_data: self.tensor_data.clone(),
        }
    }
}
