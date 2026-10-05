use nn_lib::batch_iterator::batch_iterator::DataSource;
use nn_lib::network::network::Loss;
use nn_lib::network::network::Network;
use nn_lib::tensor::tensor::Tensor;
use rayon::prelude::*;
use std::fs;

use std::time::Instant; // Замеряет время выполенения функций
// let start = Instant::now();
// println!("some code");
// let elapsed = start.elapsed();
// println!("Timer: {:?}", elapsed);

fn main() {
    // test_gradients_rust();
    // test_conv();
    // test_gradients_rust();
    train_mnist();
}

use image::ImageReader;

/// Загружает подмножество MNIST (train или test)
fn load_mnist_data(
    dir_path: &str,
    subset_size: usize,
    normalize: bool,
) -> (DataSource, DataSource) {
    let mut paths: Vec<String> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();

    if let Ok(entries) = fs::read_dir(dir_path) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "png" {
                        let path_str = path.to_string_lossy().to_string();
                        paths.push(path_str);

                        let file_name = path.file_name().unwrap().to_str().unwrap();
                        if let Some(label_str) = file_name.split('_').next() {
                            if let Ok(label) = label_str.parse::<f32>() {
                                labels.push(label);
                            }
                        }
                    }
                }
            }
        }
    }

    let total = paths.len();
    let subset_paths: Vec<String> = paths.into_iter().take(subset_size).collect();
    let subset_labels: Vec<f32> = labels.into_iter().take(subset_size).collect();

    println!("Загружено {} картинок (из {})", subset_paths.len(), total);

    // One-hot encoding (10 классов)
    let num_classes = 10;
    let mut one_hot = Vec::with_capacity(subset_labels.len() * num_classes);
    for &label in &subset_labels {
        let mut row = vec![0.0; num_classes];
        row[label as usize] = 1.0;
        one_hot.extend(row);
    }

    let x_source = DataSource::Images(subset_paths);
    let y_source = DataSource::Tensor(
        Tensor::new(
            one_hot,
            vec![subset_labels.len(), num_classes],
            false,
            vec![],
            None,
        )
        .tensor_data
        .clone(),
    );
    println!("Проведён One Hot Encode");

    (x_source, y_source)
}

/// Загружает тестовые данные в память для проверки точности
fn load_test_data(test_dir: &str) -> (Tensor, Tensor) {
    let mut paths: Vec<String> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();

    if let Ok(entries) = fs::read_dir(test_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "png" {
                        let path_str = path.to_string_lossy().to_string();
                        paths.push(path_str);

                        let file_name = path.file_name().unwrap().to_str().unwrap();
                        if let Some(label_str) = file_name.split('_').next() {
                            if let Ok(label) = label_str.parse::<f32>() {
                                labels.push(label);
                            }
                        }
                    }
                }
            }
        }
    }

    let num_samples = paths.len();
    let mut x_data = Vec::with_capacity(num_samples * 1 * 28 * 28);

    for path in &paths {
        let img = ImageReader::open(path)
            .unwrap()
            .decode()
            .unwrap()
            .to_luma8();
        let pixels: Vec<f32> = img.pixels().map(|p| p[0] as f32 / 255.0).collect();
        x_data.extend(pixels);
    }

    let x = Tensor::new(x_data, vec![num_samples, 1, 28, 28], false, vec![], None);
    let y = Tensor::new(labels, vec![num_samples, 1], false, vec![], None);

    println!("Тестовые данные загружены");
    (x, y)
}

/// Функция для тестирования MNIST
fn test_mnist() {
    // ===== 1. Загрузка данных =====
    let train_dir = "./mnist/train";
    let test_dir = "./mnist/test";

    let (x_train_source, y_train_source) = load_mnist_data(train_dir, 60_000, true);
    let (x_test, y_test) = load_test_data(test_dir);

    println!("Данные готовы");
    // ===== 2. Архитектура сети =====
    let mut net = Network::new();

    // Свёрточные слои
    let conv1 = net.Conv2d(1, 32, (3, 3));
    let conv2 = net.Conv2d(32, 64, (3, 3));
    let conv3 = net.Conv2d(64, 128, (3, 3)); // дополнительный слой

    // Полносвязные слои
    // После трёх свёрток с MaxPool: 28→26→13→11→5→3→1? Давай посчитаем
    // conv1: 28→26, MaxPool: 26→13
    // conv2: 13→11, MaxPool: 11→5
    // conv3: 5→3, MaxPool: 3→1
    // Итог: 128 * 1 * 1 = 128
    let (fc1_w, fc1_b) = net.Linear(128, 64);
    let (fc2_w, fc2_b) = net.Linear(64, 10);

    let forward_fn = |x: &Tensor| {
        x.conv2d(&conv1, (1, 1))
            .max_pool((2, 2), (2, 2))
            .relu()
            .conv2d(&conv2, (1, 1))
            .max_pool((2, 2), (2, 2))
            .relu()
            .conv2d(&conv3, (1, 1))
            .max_pool((2, 2), (2, 2))
            .relu()
            .flatten()
            .matmul(&fc1_w)
            .add(&fc1_b)
            .relu()
            .matmul(&fc2_w)
            .add(&fc2_b)
    };

    // ===== 3. Обучение =====
    println!("=== Обучение на MNIST (60000 картинок) ===");
    println!(
        "Архитектура: Conv2d(1→32) → Pool → Conv2d(32→64) → Pool → Conv2d(64→128) → Pool → Linear(128→64) → Linear(64→10)"
    );

    net.fit(
        10,    // эпохи
        0.001, // learning rate (уменьшим для стабильности)
        x_train_source,
        y_train_source,
        Loss::CrossEntropyWithSoftmax,
        64, // batch_size
        1,  // verbose
        forward_fn,
    );

    // ===== 4. Тестирование =====
    println!("\n=== Тестирование на 10000 картинках ===");

    let output = net.forward(&x_test, forward_fn);
    let predictions = output._softmax();
    let pred_data = predictions.tensor_data.borrow().data.clone();

    let mut correct = 0;
    let num_samples = x_test.tensor_data.borrow().shape[0];
    let num_classes = 10;

    for i in 0..num_samples {
        let start = i * num_classes;
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;

        for j in 0..num_classes {
            let val = pred_data[start + j];
            if val > max_val {
                max_val = val;
                max_idx = j;
            }
        }

        let target = y_test.tensor_data.borrow().data[i] as usize;
        if max_idx == target {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / num_samples as f32 * 100.0;
    println!(
        "Точность на тестовой выборке: {:.2}% ({}/{})",
        accuracy, correct, num_samples
    );
    println!("=== Обучение завершено! ===");
}

/// Загружает подмножество MNIST с нормализацией
fn load_mnist_subset(subset_size: usize) -> (DataSource, DataSource) {
    let train_dir = "./mnist/train";
    let mut paths: Vec<String> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();

    // Собираем все пути
    if let Ok(entries) = fs::read_dir(train_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "png" {
                        let path_str = path.to_string_lossy().to_string();
                        paths.push(path_str);

                        let file_name = path.file_name().unwrap().to_str().unwrap();
                        if let Some(label_str) = file_name.split('_').next() {
                            if let Ok(label) = label_str.parse::<f32>() {
                                labels.push(label);
                            }
                        }
                    }
                }
            }
        }
    }

    let total = paths.len();
    let subset_paths: Vec<String> = paths.into_iter().take(subset_size).collect();
    let subset_labels: Vec<f32> = labels.into_iter().take(subset_size).collect();

    println!("Загружено {} картинок (из {})", subset_paths.len(), total);

    // Параметры нормализации (из PyTorch)
    let mean = 0.1307;
    let std = 0.3081;

    // Параллельная загрузка и нормализация картинок
    let x_data: Vec<f32> = subset_paths
        .par_iter()
        .flat_map(|path| {
            let img = ImageReader::open(path)
                .unwrap()
                .decode()
                .unwrap()
                .to_luma8();
            img.pixels()
                .map(|p| (p[0] as f32 / 255.0 - mean) / std)
                .collect::<Vec<f32>>()
        })
        .collect();

    let num_samples = subset_paths.len();
    let x = Tensor::new(x_data, vec![num_samples, 1, 28, 28], false, vec![], None);
    let x_source = DataSource::Tensor(x.tensor_data.clone());

    // One-hot encoding (10 классов)
    let num_classes = 10;
    let mut one_hot = Vec::with_capacity(subset_labels.len() * num_classes);
    for &label in &subset_labels {
        let mut row = vec![0.0; num_classes];
        row[label as usize] = 1.0;
        one_hot.extend(row);
    }

    let y = Tensor::new(
        one_hot,
        vec![subset_labels.len(), num_classes],
        false,
        vec![],
        None,
    );
    let y_source = DataSource::Tensor(y.tensor_data.clone());

    (x_source, y_source)
}

fn train_mnist() {
    let (x_source, y_source) = load_mnist_subset(60_000);

    // Создаём сеть
    let mut net = Network::new();

    // Архитектура (3 свёрточных слоя)
    let conv1 = net.Conv2d(1, 32, (3, 3));
    let conv2 = net.Conv2d(32, 64, (3, 3));
    let conv3 = net.Conv2d(64, 128, (3, 3));
    let (fc1_w, fc1_b) = net.Linear(128 * 3 * 3, 256);
    let (fc2_w, fc2_b) = net.Linear(256, 128);
    let (fc3_w, fc3_b) = net.Linear(128, 10);

    let forward_fn = |x: &Tensor| {
        let pipa = x
            .pad((1, 1, 1, 1))
            .conv2d(&conv1, (1, 1))
            .max_pool((2, 2), (2, 2))
            .relu()
            .pad((1, 1, 1, 1))
            .conv2d(&conv2, (1, 1))
            .max_pool((2, 2), (2, 2))
            .relu()
            .pad((1, 1, 1, 1))
            .conv2d(&conv3, (1, 1))
            .max_pool((2, 2), (2, 2))
            .relu();
        let biba = pipa
            .flatten()
            .matmul(&fc1_w)
            .add(&fc1_b)
            .relu()
            .matmul(&fc2_w)
            .add(&fc2_b)
            .relu()
            .matmul(&fc3_w)
            .add(&fc3_b);
        biba
    };

    println!("\n=== Обучение на MNIST (10000 картинок) ===");

    net.fit(
        10,     // эпохи
        0.001, // learning rate
        x_source,
        y_source,
        Loss::CrossEntropyWithSoftmax,
        64, // batch_size
        1,  // verbose
        forward_fn,
    );

    println!("\n=== Обучение завершено! ===");
}

fn test_gradients_rust() {
    let x = Tensor::new(
        (0..3 * 4 * 4).map(|f| f as f32 / 10.).collect(),
        vec![1, 3, 4, 4],
        true,
        vec![],
        None,
    );
    // pad 1
    let conv1 = Tensor::new(
        (0..9 * 3).map(|f| f as f32 / 10.).collect(),
        vec![1, 3, 3, 3],
        true,
        vec![],
        None,
    );

    //relu
    //pool
    //pad 1
    let conv2 = Tensor::new(
        (0..9).map(|f| f as f32 / 10.).collect(),
        vec![1, 1, 3, 3],
        true,
        vec![],
        None,
    );
    //relu
    //pool
    //flatten -> 4 1?

    let w1 = Tensor::new(vec![1.5 / 10.], vec![1, 1], true, vec![], None);
    let b1 = Tensor::new(vec![1.5 / 10.], vec![1, 1], true, vec![], None);
    // ===== 1. Свёрточная часть =====
    println!("=== Свёрточная часть ===");

    // Шаг 1: Паддинг входа
    let padded1 = x.pad((1, 1, 1, 1));
    println!("1. padded1 (после pad):\n{}", padded1);

    // Шаг 2: Первая свёртка
    let conv1_out = padded1.conv2d(&conv1, (1, 1));
    println!("2. conv1_out (после conv1):\n{}", conv1_out);

    // Шаг 3: ReLU
    let relu1_out = conv1_out.relu();
    println!("3. relu1_out (после relu):\n{}", relu1_out);

    // Шаг 4: MaxPool
    let pool1_out = relu1_out.max_pool((2, 2), (2, 2));
    println!("4. pool1_out (после max_pool):\n{}", pool1_out);

    // Шаг 5: Паддинг
    let padded2 = pool1_out.pad((1, 1, 1, 1));
    println!("5. padded2 (после pad):\n{}", padded2);

    // Шаг 6: Вторая свёртка
    let conv2_out = padded2.conv2d(&conv2, (1, 1));
    println!("6. conv2_out (после conv2):\n{}", conv2_out);

    // Шаг 7: ReLU
    let relu2_out = conv2_out.relu();
    println!("7. relu2_out (после relu):\n{}", relu2_out);

    // Шаг 8: MaxPool
    let pool2_out = relu2_out.max_pool((2, 2), (2, 2));
    println!("8. pool2_out (после max_pool):\n{}", pool2_out);

    // Шаг 9: Flatten
    let flat = pool2_out.flatten();
    println!("9. flat (после flatten):\n{}", flat);

    println!("\n=== Полносвязная часть ===");

    // ===== 2. Полносвязная часть =====
    // Шаг 10: Первый линейный слой
    let fc1_out = flat.matmul(&w1).add(&b1);
    println!("10. fc1_out (после w1*x + b1):\n{}", fc1_out);

    // Шаг 11: ReLU
    let output = fc1_out.relu();
    println!("11. relu3_out (после relu):\n{}", output);

    // ===== 3. Loss =====
    let target = Tensor::new(vec![1.0], vec![1, 1], false, vec![], None);
    let loss = output.mse(&target);
    println!("\nLoss: {}", loss.tensor_data.borrow().data[0]);

    // ===== 4. Backward =====
    loss.backward();

    // ===== 5. Градиенты =====
    println!("\n=== Градиенты ===");
    println!("conv1 grad: {:?}", conv1.tensor_data.borrow().grad);
    println!("conv2 grad: {:?}", conv2.tensor_data.borrow().grad);
    println!("w1 grad: {:?}", w1.tensor_data.borrow().grad);
    println!("b1 grad: {:?}", b1.tensor_data.borrow().grad);
}

fn test_conv() {
    let x = Tensor::new(
        (0..3 * 4 * 5).map(|f| f as f32 / 10.).collect(),
        vec![1, 3, 4, 5],
        true,
        vec![],
        None,
    );
    let conv1 = Tensor::new(
        (0..9 * 3).map(|f| f as f32 / 10.).collect(),
        vec![1, 3, 3, 3],
        true,
        vec![],
        None,
    );

    let output = x.conv2d(&conv1, (1, 1));
    println!("output: {}", output);
    output.tensor_data.borrow_mut().grad = vec![2., 3., 4., 5., 6., 7.];
    output.backward();
    println!("{}", x.grad());
    println!("{}", conv1.grad());
}
