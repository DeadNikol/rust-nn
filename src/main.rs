use nn_lib::batch_iterator::data_source::DataSource;
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
    // train_mnist();
    train_mnist_new();
    // pipa();
}

fn pipa() {
    let a = Tensor::from_path("src/batch_iterator/images/0_000000.png".to_string()).unwrap();
    println!("{}", a);
}

use image::ImageReader;

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
        10,    // эпохи
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

fn train_mnist_new() {
    // ===== 1. Путь к MNIST =====
    let mnist_root = r"C:\Папка для всего\Папка для VS Code\rust\NN\mnist\train";

    // ===== 2. Папки-классы: train/0/, train/1/, ..., train/9/ =====
    let folders: Vec<(String, usize)> = (0..10)
        .map(|i| (format!("{}/{}", mnist_root, i), i))
        .collect();

    // ===== 3. Загружаем пути и метки =====
    let (x_source_raw, labels) = DataSource::from_folders_with_classes(folders);

    println!("Загружено {} картинок", x_source_raw.len());
    assert!(x_source_raw.len() > 0, "MNIST не загрузился");

    // ===== 4. Нормализация (MNIST-стандарт) =====
    let x_source = x_source_raw.normalized(0.1307, 0.3081);

    // ===== 5. One-hot метки =====
    let y_source = DataSource::one_hot(&labels, 10);

    // ===== 6. Сеть =====
    let mut net = Network::new();

    // Архитектура (3 свёрточных слоя)
    let conv1 = net.Conv2d(1, 32, (3, 3));
    let conv2 = net.Conv2d(32, 64, (3, 3));
    let conv3 = net.Conv2d(64, 128, (3, 3));
    let (fc1_w, fc1_b) = net.Linear(128 * 3 * 3, 256);
    let (fc2_w, fc2_b) = net.Linear(256, 128);
    let (fc3_w, fc3_b) = net.Linear(128, 10);

    let forward_fn = |x: &Tensor| {
        macro_rules! time_it {
            ($label:expr, $body:expr) => {{
                let t = Instant::now();
                let result = $body;
                println!("  [fwd] {:>12}: {:>10.3?}", $label, t.elapsed());
                result
            }};
        }

        let pipa = time_it!("pad1", x.pad((1, 1, 1, 1)));
        let pipa = time_it!("conv1", pipa.conv2d(&conv1, (1, 1)));
        let pipa = time_it!("pool1", pipa.max_pool((2, 2), (2, 2)));
        let pipa = time_it!("relu1", pipa.relu());

        let pipa = time_it!("pad2", pipa.pad((1, 1, 1, 1)));
        let pipa = time_it!("conv2", pipa.conv2d(&conv2, (1, 1)));
        let pipa = time_it!("pool2", pipa.max_pool((2, 2), (2, 2)));
        let pipa = time_it!("relu2", pipa.relu());

        let pipa = time_it!("pad3", pipa.pad((1, 1, 1, 1)));
        let pipa = time_it!("conv3", pipa.conv2d(&conv3, (1, 1)));
        let pipa = time_it!("pool3", pipa.max_pool((2, 2), (2, 2)));
        let pipa = time_it!("relu3", pipa.relu());

        let flat = time_it!("flatten", pipa.flatten());
        let fc1 = time_it!("fc1_mm", flat.matmul(&fc1_w));
        let fc1 = time_it!("fc1_add", fc1.add(&fc1_b));
        let fc1 = time_it!("fc1_relu", fc1.relu());
        let fc2 = time_it!("fc2_mm", fc1.matmul(&fc2_w));
        let fc2 = time_it!("fc2_add", fc2.add(&fc2_b));
        let fc2 = time_it!("fc2_relu", fc2.relu());
        let fc3 = time_it!("fc3_mm", fc2.matmul(&fc3_w));
        let fc3 = time_it!("fc3_add", fc3.add(&fc3_b));
        fc3
    };

    // ===== 7. Обучение =====
    println!("=== Обучение на MNIST ===");

    net.fit(
        1,    // эпохи
        0.001, // lr
        x_source,
        y_source,
        Loss::CrossEntropyWithSoftmax,
        64, // batch_size
        1,  // verbose = 1 (печать каждую эпоху)
        forward_fn,
    );

    println!("=== Обучение завершено ===");
}
