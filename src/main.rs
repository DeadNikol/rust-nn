mod tensor;
use std::io::ErrorKind::OutOfMemory;

use tensor::*;

use crate::network::Network;

mod batch_iterator;
mod graph;
mod layers;
mod network;
mod optims;

#[cfg(test)]
mod unit_tests;

// Я для matmul жестоко нахардкодил размерность 2, однако в этом же matmul я писал, что размерность может быть любой, лишь бы пер

fn main() {
    fn test_iris() {
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        fn load_iris_data(file_path: &str) -> (Tensor, Tensor) {
            let file = File::open(file_path).expect("Файл не найден");
            let reader = BufReader::new(file);

            let mut x_data = Vec::new(); // все признаки
            let mut y_data = Vec::new(); // one-hot метки

            for line in reader.lines() {
                let line = line.expect("Ошибка чтения строки");
                if line.starts_with("Id") || line.trim().is_empty() {
                    continue; // пропускаем заголовок и пустые строки
                }

                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() < 6 {
                    continue;
                }

                // Парсим признаки (Id игнорируем, берём колонки 1-4)
                let sepal_len: f32 = parts[1].parse().unwrap();
                let sepal_wid: f32 = parts[2].parse().unwrap();
                let petal_len: f32 = parts[3].parse().unwrap();
                let petal_wid: f32 = parts[4].parse().unwrap();

                x_data.push(sepal_len);
                x_data.push(sepal_wid);
                x_data.push(petal_len);
                x_data.push(petal_wid);

                // Парсим класс (колонка 5)
                let species = parts[5].trim().to_lowercase();
                let class_index = match species.as_str() {
                    "iris-setosa" => 0,
                    "iris-versicolor" => 1,
                    "iris-virginica" => 2,
                    _ => continue,
                };

                // Создаём one-hot вектор из 3 элементов
                let mut one_hot = vec![0.0, 0.0, 0.0];
                one_hot[class_index] = 1.0;
                y_data.extend(one_hot);
            }

            let num_samples = x_data.len() / 4;

            let x = Tensor::new(x_data, vec![num_samples, 4], false, vec![], None);
            let y = Tensor::new(y_data, vec![num_samples, 3], false, vec![], None);

            (x, y)
        }

        fn calculate_accuracy(predictions: &Tensor, targets: &Tensor) -> f32 {
            let batch_size = predictions.tensor_data.borrow().shape[0];
            let num_classes = predictions.tensor_data.borrow().shape[1];

            let pred_data = predictions.tensor_data.borrow();
            let target_data = targets.tensor_data.borrow();

            let mut correct = 0;

            for i in 0..batch_size {
                let start = i * num_classes;

                // Находим предсказанный класс (индекс максимальной вероятности)
                let mut pred_class = 0;
                let mut max_prob = pred_data.data[start];
                for j in 1..num_classes {
                    let prob = pred_data.data[start + j];
                    if prob > max_prob {
                        max_prob = prob;
                        pred_class = j;
                    }
                }

                // Находим истинный класс (индекс, где target == 1.0)
                let mut true_class = 0;
                for j in 0..num_classes {
                    if target_data.data[start + j] == 1.0 {
                        true_class = j;
                        break;
                    }
                }

                if pred_class == true_class {
                    correct += 1;
                }
            }

            correct as f32 / batch_size as f32
        }

        let (x, y) = load_iris_data("Iris.csv");

        let mut net = network::Network::new();
        let (w1, b1) = net.Linear(4, 64);
        let (w2, b2) = net.Linear(64, 3);

        let forward_fn = |pipa: &Tensor| pipa.matmul(&w1).add(&b1).relu().matmul(&w2).add(&b2);

        net.fit(
            5000,
            0.0001,
            x.clone(),
            y.clone(),
            network::Loss::CrossEntropyWithSoftmax,
            150,
            1000,
            forward_fn,
        );
        let output = net.forward(&x, forward_fn)._softmax();
        let acc = calculate_accuracy(&output, &y);
        assert!(acc > 0.95, "Ожидаемая точность: 0.95. Получено: {}", acc);
        println!("accuracy: {}", acc);
    }
    test_iris();
}
