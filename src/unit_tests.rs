//! крейт для тестов всех функций. Если всё работает, значит изменения не нарушили логики
use itertools::iproduct;

use crate::{network, tensor::*};

#[test]
fn test_create_tensor() {
    let tensor = Tensor::uniform(0., 1., vec![1, 30, 20], false);
    for i in tensor.tensor_data.borrow().data.iter() {
        assert!(*i >= 0. && *i <= 1., "value {} is out of range [0., 1.]", i);
    }
}

#[test]
fn test_matmul() {
    let data1: Vec<f32> = (0..200).map(|x| x as f32).collect(); // 10*20 = 200 элементов
    let a = Tensor::new(data1, vec![1, 10, 20], false, vec![], None);

    let data2: Vec<f32> = (0..200).map(|x| x as f32).collect(); // 20*10 = 200 элементов
    let b = Tensor::new(data2, vec![1, 20, 10], false, vec![], None);

    let c = a.matmul(&b);

    assert_eq!(
        c.tensor_data.borrow().shape,
        vec![1, 10, 10],
        "При векторном умножении не сошлись размерности"
    );

    let true_data = vec![
        //Умножилось numpy
        24700.0, 24890.0, 25080.0, 25270.0, 25460.0, 25650.0, 25840.0, 26030.0, 26220.0, 26410.0,
        62700.0, 63290.0, 63880.0, 64470.0, 65060.0, 65650.0, 66240.0, 66830.0, 67420.0, 68010.0,
        100700.0, 101690.0, 102680.0, 103670.0, 104660.0, 105650.0, 106640.0, 107630.0, 108620.0,
        109610.0, 138700.0, 140090.0, 141480.0, 142870.0, 144260.0, 145650.0, 147040.0, 148430.0,
        149820.0, 151210.0, 176700.0, 178490.0, 180280.0, 182070.0, 183860.0, 185650.0, 187440.0,
        189230.0, 191020.0, 192810.0, 214700.0, 216890.0, 219080.0, 221270.0, 223460.0, 225650.0,
        227840.0, 230030.0, 232220.0, 234410.0, 252700.0, 255290.0, 257880.0, 260470.0, 263060.0,
        265650.0, 268240.0, 270830.0, 273420.0, 276010.0, 290700.0, 293690.0, 296680.0, 299670.0,
        302660.0, 305650.0, 308640.0, 311630.0, 314620.0, 317610.0, 328700.0, 332090.0, 335480.0,
        338870.0, 342260.0, 345650.0, 349040.0, 352430.0, 355820.0, 359210.0, 366700.0, 370490.0,
        374280.0, 378070.0, 381860.0, 385650.0, 389440.0, 393230.0, 397020.0, 400810.0,
    ];
    assert_eq!(
        c.tensor_data.borrow().data,
        true_data,
        "Неправильное умножение"
    )
}

#[test]
fn test_transpose_20x10() {
    // Тест генерировал синий кит
    let rows = 20;
    let cols = 10;
    let data: Vec<f32> = (0..rows * cols).map(|x| x as f32).collect();
    let a = Tensor::new(data, vec![1, rows, cols], false, vec![], None);
    let output = a.transpose();

    // Правильный расчёт ожидаемого результата. Я так и не понял, почему это работает
    let expected: Vec<f32> = (0..cols)
        .flat_map(|j| (0..rows).map(move |i| (i * cols + j) as f32))
        .collect();

    assert_eq!(output.tensor_data.borrow().data, expected);
    assert_eq!(output.tensor_data.borrow().shape, vec![1, cols, rows]);
}

#[test]
fn test_add() {
    // ---------- 1. Обычное сложение (одинаковые формы) ----------
    let data: Vec<f32> = (0..10).map(|x| x as f32).collect(); // 2×5 = 10 элементов
    let a = Tensor::new(data.clone(), vec![1, 2, 5], false, vec![], None);
    let output = a.add(&a);
    let expected: Vec<f32> = data.iter().map(|&x| x + x).collect();
    assert_eq!(
        output.tensor_data.borrow().data,
        expected,
        "Обычное сложение двух матриц 2×5 дало неверный результат"
    );

    // ---------- 2. Сложение матрицы [1, 3, 4] с bias-строкой [1, 1, 4] ----------
    let rows = 3;
    let cols = 4;
    // Матрица: значения 0..11 (3*4)
    let mat_data: Vec<f32> = (0..(rows * cols)).map(|x| x as f32).collect();
    // Bias: строка из 4 элементов, например 10.0, 20.0, 30.0, 40.0
    let bias_data: Vec<f32> = (0..cols).map(|x| (x + 1) as f32 * 10.0).collect();

    let matrix = Tensor::new(mat_data.clone(), vec![1, rows, cols], false, vec![], None);
    let bias = Tensor::new(bias_data.clone(), vec![1, 1, cols], false, vec![], None);

    let result = matrix.add(&bias); // matrix + bias

    // Ожидаемый результат: каждая строка mat_data плюс bias_data
    let mut expected_bcast = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            expected_bcast.push(mat_data[i * cols + j] + bias_data[j]);
        }
    }
    assert_eq!(
        result.tensor_data.borrow().data,
        expected_bcast,
        "Matrix + bias broadcasting не работает"
    );

    // ---------- 3. Симметричный случай: bias + matrix ----------
    let result2 = bias.add(&matrix); // bias + matrix
    assert_eq!(
        result2.tensor_data.borrow().data,
        expected_bcast,
        "bias + matrix даёт другой результат (должно быть симметрично)"
    );
}

#[test]
fn test_matmul_backward() {
    // Этот код писал я, asserq_eq писал синий кит по моим выводам. Они совпали с pyTorch
    let a = Tensor::new(
        (0..10).map(|f| f as f32).collect(),
        vec![1, 2, 5],
        false,
        vec![],
        None,
    );
    let b = Tensor::new(
        (0..10).map(|f| f as f32).collect(),
        vec![1, 5, 2],
        false,
        vec![],
        None,
    );
    let c = Tensor::new(
        (0..2).map(|f| f as f32).collect(),
        vec![1, 2, 1],
        false,
        vec![],
        None,
    );
    let output = a.matmul(&b).matmul(&c);

    output.backward();

    // Ожидаемые градиенты для a
    let expected_a: Vec<f32> = vec![1.0, 3.0, 5.0, 7.0, 9.0, 1.0, 3.0, 5.0, 7.0, 9.0];
    assert_eq!(
        a.tensor_data.borrow().grad,
        expected_a,
        "Градиенты для a не совпадают с ожидаемыми"
    );

    // Ожидаемые градиенты для b
    let expected_b: Vec<f32> = vec![0.0, 5.0, 0.0, 7.0, 0.0, 9.0, 0.0, 11.0, 0.0, 13.0];
    assert_eq!(
        b.tensor_data.borrow().grad,
        expected_b,
        "Градиенты для b не совпадают с ожидаемыми"
    );

    // Ожидаемые градиенты для c
    let expected_c: Vec<f32> = vec![220.0, 265.0];
    assert_eq!(
        c.tensor_data.borrow().grad,
        expected_c,
        "Градиенты для c не совпадают с ожидаемыми"
    );
}

#[test]
fn test_matmul_add_backward() {
    let a = Tensor::new(
        (0..10).map(|f| f as f32).collect(),
        vec![1, 2, 5],
        false,
        vec![],
        None,
    );
    let b = Tensor::new(
        (0..10).map(|f| f as f32).collect(),
        vec![1, 5, 2],
        false,
        vec![],
        None,
    );
    let c = Tensor::new(
        (0..4).map(|f| f as f32).collect(),
        vec![1, 2, 2],
        false,
        vec![],
        None,
    );
    let output = a.matmul(&b).add(&c);

    output.backward();

    // Ожидаемые градиенты для a
    let expected_a: Vec<f32> = vec![1.0, 5.0, 9.0, 13.0, 17.0, 1.0, 5.0, 9.0, 13.0, 17.0];
    assert_eq!(
        a.tensor_data.borrow().grad,
        expected_a,
        "Градиенты для a не совпадают с ожидаемыми"
    );

    // Ожидаемые градиенты для b
    let expected_b: Vec<f32> = vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0, 11.0, 11.0, 13.0, 13.0];
    assert_eq!(
        b.tensor_data.borrow().grad,
        expected_b,
        "Градиенты для b не совпадают с ожидаемыми"
    );

    // Ожидаемые градиенты для c
    let expected_c: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    assert_eq!(
        c.tensor_data.borrow().grad,
        expected_c,
        "Градиенты для c не совпадают с ожидаемыми"
    );
}

#[test]
fn test_relu_backward() {
    let a = Tensor::new(
        (0..10).map(|f| f as f32).collect(),
        vec![1, 2, 5],
        false,
        vec![],
        None,
    );
    let b = Tensor::new(
        (0..10).map(|f| f as f32).collect(),
        vec![1, 5, 2],
        false,
        vec![],
        None,
    );
    b.tensor_data.borrow_mut().data[9] = -9.;
    let c = Tensor::new(
        (0..2).map(|f| f as f32).collect(),
        vec![1, 2, 1],
        false,
        vec![],
        None,
    );
    let output = a.matmul(&b).relu().matmul(&c);

    output.backward();

    // Ожидаемые градиенты для a
    let expected_a: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 5.0, 7.0, -9.0];
    assert_eq!(
        a.tensor_data.borrow().grad,
        expected_a,
        "Градиенты для a не совпадают с ожидаемыми"
    );

    // Ожидаемые градиенты для b
    let expected_b: Vec<f32> = vec![0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0, 9.0];
    assert_eq!(
        b.tensor_data.borrow().grad,
        expected_b,
        "Градиенты для b не совпадают с ожидаемыми"
    );

    // Ожидаемые градиенты для c
    let expected_c: Vec<f32> = vec![220.0, 33.0];
    assert_eq!(
        c.tensor_data.borrow().grad,
        expected_c,
        "Градиенты для c не совпадают с ожидаемыми"
    );
}

#[test]
fn test_like_python() {
    let mut net = network::Network::new();
    let (w1, b1) = net.Linear(6, 10);
    let (w2, b2) = net.Linear(10, 1);

    // Данные
    let x: Vec<f32> = iproduct!([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
        .flat_map(|(a, b, c, d, e, f)| {
            vec![a as f32, b as f32, c as f32, d as f32, e as f32, f as f32]
        })
        .collect();

    let y: Vec<f32> = x
        .chunks(6)
        .filter_map(|chunk| chunk.get(2).copied()) // третий бит (индекс 2)
        .collect();

    let x = Tensor::new(x.clone(), vec![1, x.len() / 6, 6], false, vec![], None);
    let y = Tensor::new(y.clone(), vec![1, y.len(), 1], false, vec![], None);

    let forward_fn = |x: &Tensor| x.matmul(&w1).add(&b1).tanh().matmul(&w2).add(&b2).tanh();

    // let start = Instant::now();
    net.fit(
        10000,
        0.001,
        x.clone(),
        y.clone(),
        network::Loss::MSE,
        150,
        0,
        forward_fn,
    );
    // let duration = start.elapsed();

    // println!("\n=== Training Time ===");
    // println!("Duration: {:?}", duration);

    // Вычисляем финальную MSE
    let output = forward_fn(&x);
    let final_loss = output.mse(&y);
    // println!("Final MSE: {}", final_loss.tensor_data.borrow().data[0]);
    assert!(
        final_loss.tensor_data.borrow().data[0] <= 0.02,
        "MSE {} больше приемлемого значения 0.02",
        final_loss.tensor_data.borrow().data[0]
    );

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
    assert!(
        accuracy >= 0.98,
        "Точность {} меньше приемлемого значения 0.98",
        accuracy
    );

    // println!("Accuracy: {:.4}", accuracy);
}

#[test]
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

        let x = Tensor::new(x_data, vec![1, num_samples, 4], false, vec![], None);
        let y = Tensor::new(y_data, vec![1, num_samples, 3], false, vec![], None);

        (x, y)
    }

    fn calculate_accuracy(predictions: &Tensor, targets: &Tensor) -> f32 {
        let batch_size = predictions.tensor_data.borrow().shape[1];
        let num_classes = predictions.tensor_data.borrow().shape[2];

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
        0.005,
        x.clone(),
        y.clone(),
        network::Loss::CrossEntropyWithSoftmax,
        150,
        0,
        forward_fn,
    );
    let output = net.forward(&x, forward_fn)._softmax();
    let acc = calculate_accuracy(&output, &y);
    assert!(acc > 0.95, "Ожидаемая точность: 0.95. Получено: {}", acc);
    // println!("accuracy: {}", acc);
}
