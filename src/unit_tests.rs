//! крейт для тестов всех функций. Если всё работает, значит изменения не нарушили логики
use itertools::iproduct;

use crate::{
    batch_iterator::DataSource,
    network::{self, Loss},
    tensor::*,
};

#[test]
fn test_create_tensor() {
    let tensor = Tensor::uniform(0., 1., vec![30, 20], false);
    for i in tensor.tensor_data.borrow().data.iter() {
        assert!(*i >= 0. && *i <= 1., "value {} is out of range [0., 1.]", i);
    }
}

#[test]
fn test_matmul() {
    let data1: Vec<f32> = (0..200).map(|x| x as f32).collect(); // 10*20 = 200 элементов
    let a = Tensor::new(data1, vec![10, 20], false, vec![], None);

    let data2: Vec<f32> = (0..200).map(|x| x as f32).collect(); // 20*10 = 200 элементов
    let b = Tensor::new(data2, vec![20, 10], false, vec![], None);

    let c = a.matmul(&b);

    assert_eq!(
        c.tensor_data.borrow().shape,
        vec![10, 10],
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
    let rows = 20;
    let cols = 10;
    let data: Vec<f32> = (0..rows * cols).map(|x| x as f32).collect();
    let a = Tensor::new(data, vec![rows, cols], false, vec![], None);
    let output = a.transpose();

    let expected: Vec<f32> = (0..cols)
        .flat_map(|j| (0..rows).map(move |i| (i * cols + j) as f32))
        .collect();

    assert_eq!(output.tensor_data.borrow().data, expected);
    assert_eq!(output.tensor_data.borrow().shape, vec![cols, rows]);
}

#[test]
fn test_add() {
    // ---------- 1. Обычное сложение (одинаковые формы) ----------
    let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
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
    let mat_data: Vec<f32> = (0..(rows * cols)).map(|x| x as f32).collect();
    let bias_data: Vec<f32> = (0..cols).map(|x| (x + 1) as f32 * 10.0).collect();

    let matrix = Tensor::new(mat_data.clone(), vec![1, rows, cols], false, vec![], None);
    let bias = Tensor::new(bias_data.clone(), vec![1, 1, cols], false, vec![], None);

    let result = matrix.add(&bias);

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
}

#[test]
fn test_matmul_backward() {
    let a = Tensor::new(
        (0..10).map(|f| f as f32).collect(),
        vec![2, 5],
        true,
        vec![],
        None,
    );
    let b = Tensor::new(
        (0..10).map(|f| f as f32).collect(),
        vec![5, 2],
        true,
        vec![],
        None,
    );
    let c = Tensor::new(
        (0..2).map(|f| f as f32).collect(),
        vec![2, 1],
        true,
        vec![],
        None,
    );
    let output = a.matmul(&b).matmul(&c);
    output.backward();

    let expected_a: Vec<f32> = vec![1.0, 3.0, 5.0, 7.0, 9.0, 1.0, 3.0, 5.0, 7.0, 9.0];
    assert_eq!(a.tensor_data.borrow().grad, expected_a);

    let expected_b: Vec<f32> = vec![0.0, 5.0, 0.0, 7.0, 0.0, 9.0, 0.0, 11.0, 0.0, 13.0];
    assert_eq!(b.tensor_data.borrow().grad, expected_b);

    let expected_c: Vec<f32> = vec![220.0, 265.0];
    assert_eq!(c.tensor_data.borrow().grad, expected_c);
}

#[test]
fn test_matmul_add_backward() {
    let a = Tensor::new(
        (0..10).map(|f| f as f32).collect(),
        vec![2, 5],
        true,
        vec![],
        None,
    );
    let b = Tensor::new(
        (0..10).map(|f| f as f32).collect(),
        vec![5, 2],
        true,
        vec![],
        None,
    );
    let c = Tensor::new(
        (0..4).map(|f| f as f32).collect(),
        vec![2, 2],
        true,
        vec![],
        None,
    );
    let output = a.matmul(&b).add(&c);
    output.backward();

    let expected_a: Vec<f32> = vec![1.0, 5.0, 9.0, 13.0, 17.0, 1.0, 5.0, 9.0, 13.0, 17.0];
    assert_eq!(a.tensor_data.borrow().grad, expected_a);

    let expected_b: Vec<f32> = vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0, 11.0, 11.0, 13.0, 13.0];
    assert_eq!(b.tensor_data.borrow().grad, expected_b);

    let expected_c: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    assert_eq!(c.tensor_data.borrow().grad, expected_c);
}

#[test]
fn test_relu_backward() {
    let a = Tensor::new(
        (0..10).map(|f| f as f32).collect(),
        vec![2, 5],
        true,
        vec![],
        None,
    );
    let b = Tensor::new(
        (0..10).map(|f| f as f32).collect(),
        vec![5, 2],
        true,
        vec![],
        None,
    );
    b.tensor_data.borrow_mut().data[9] = -9.;
    let c = Tensor::new(
        (0..2).map(|f| f as f32).collect(),
        vec![2, 1],
        true,
        vec![],
        None,
    );
    let output = a.matmul(&b).relu().matmul(&c);
    output.backward();

    let expected_a: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 5.0, 7.0, -9.0];
    assert_eq!(a.tensor_data.borrow().grad, expected_a);

    let expected_b: Vec<f32> = vec![0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0, 9.0];
    assert_eq!(b.tensor_data.borrow().grad, expected_b);

    let expected_c: Vec<f32> = vec![220.0, 33.0];
    assert_eq!(c.tensor_data.borrow().grad, expected_c);
}

#[test]
fn test_conv2d_backward() {
    let x = Tensor::new(
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![1, 1, 3, 3],
        false,
        vec![],
        None,
    );
    let kernel = Tensor::new(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1, 1, 2, 2],
        false,
        vec![],
        None,
    );

    let output = x.conv2d(&kernel, (1, 1));
    // Ожидаемый результат: [[19, 25], [37, 43]]
    assert_eq!(
        output.tensor_data.borrow().data,
        vec![19.0, 25.0, 37.0, 43.0]
    );
}

#[test]
fn test_padding() {
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
        false,
        vec![],
        None,
    );

    let padded = x.pad((1, 1, 1, 1));

    // Ожидаем: [1, 1, 5, 5] с нулями по краям
    let expected = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 7.0, 8.0,
        9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    assert_eq!(padded.tensor_data.borrow().data, expected);
    assert_eq!(padded.tensor_data.borrow().shape, vec![1, 1, 5, 5]);
}

// ===== 2. Тест свёртки с паддингом =====
#[test]
fn test_conv2d_with_padding() {
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
        false,
        vec![],
        None,
    );

    let kernel = Tensor::new(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1, 1, 2, 2],
        false,
        vec![],
        None,
    );

    // Свёртка с паддингом 1
    let output = x.pad((1, 1, 1, 1)).conv2d(&kernel, (1, 1));

    // Ожидаемый результат (проверим через Python)
    let expected = vec![
        3.0, 8.0, 13.0, 6.0, 13.0, 25.0, 31.0, 12.0, 25.0, 43.0, 49.0, 18.0, 7.0, 8.0, 9.0, 0.0,
    ];

    assert_eq!(output.tensor_data.borrow().data, expected);
    assert_eq!(output.tensor_data.borrow().shape, vec![1, 1, 4, 4]);
}

// ===== 3. Тест MaxPool =====
#[test]
fn test_max_pool() {
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
        false,
        vec![],
        None,
    );

    let output = x.max_pool((2, 2), (1, 1));

    // Ожидаем: максимумы в каждом окне 2×2
    let expected = vec![5.0, 6.0, 8.0, 9.0];

    assert_eq!(output.tensor_data.borrow().data, expected);
    assert_eq!(output.tensor_data.borrow().shape, vec![1, 1, 2, 2]);
}

// ===== 4. Тест свёртка + MaxPool =====
#[test]
fn test_conv2d_maxpool() {
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
        false,
        vec![],
        None,
    );

    let kernel = Tensor::new(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1, 1, 2, 2],
        false,
        vec![],
        None,
    );

    let conv_out = x.conv2d(&kernel, (1, 1));
    let pool_out = conv_out.max_pool((2, 2), (1, 1));

    // conv_out: [[19, 25], [37, 43]]
    // maxpool 2×2, stride=1: максимум из всех 2×2 окон
    let expected = vec![49.0];

    assert_eq!(pool_out.tensor_data.borrow().data, expected);
    assert_eq!(pool_out.tensor_data.borrow().shape, vec![1, 1, 1, 1]);
}

// ===== 5. Тест последовательных свёрток =====
#[test]
fn test_two_convs() {
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
        false,
        vec![],
        None,
    );

    let kernel1 = Tensor::new(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1, 1, 2, 2],
        false,
        vec![],
        None,
    );

    let kernel2 = Tensor::new(
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1, 1, 2, 2],
        false,
        vec![],
        None,
    );

    let out1 = x.conv2d(&kernel1, (1, 1));
    let out2 = out1.conv2d(&kernel2, (1, 1));

    // out1: [[19, 25], [37, 43]]
    // out2: 19*1 + 25*0 + 37*0 + 43*1 = 62
    let expected = vec![74.0];

    assert_eq!(out2.tensor_data.borrow().data, expected);
    assert_eq!(out2.tensor_data.borrow().shape, vec![1, 1, 1, 1]);
}

// ===== 1. Производная паддинга =====
#[test]
fn test_padding_backward() {
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
        true,
        vec![],
        None,
    );

    let padded = x.pad((1, 1, 1, 1));
    println!("{}", padded);
    padded.tensor_data.borrow_mut().grad = (0..25).map(|f| f as f32).collect();
    println!("{}", padded.grad());
    padded.backward();

    // Градиент паддинга: градиент идёт только в оригинальные пиксели
    let expected_grad = vec![6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 16.0, 17.0, 18.0];

    assert_eq!(x.tensor_data.borrow().grad, expected_grad);
}

// ===== 2. Производная свёртки с паддингом =====
#[test]
fn test_conv2d_padding_backward() {
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
        true,
        vec![],
        None,
    );

    let kernel = Tensor::new(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1, 1, 2, 2],
        true,
        vec![],
        None,
    );

    let output = x.pad((1, 1, 1, 1)).conv2d(&kernel, (1, 1));
    output.backward();

    // Проверяем, что градиенты не нулевые
    let grad_x = x.tensor_data.borrow().grad.clone();
    let grad_kernel = kernel.tensor_data.borrow().grad.clone();

    let expected_grad_x = vec![6.0; 9];
    let expected_grad_kernel = vec![45.0; 4];
    assert_eq!(grad_x, expected_grad_x);
    assert_eq!(grad_kernel, expected_grad_kernel);
}

// ===== 3. Производная MaxPool =====
#[test]
fn test_max_pool_backward() {
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
        true,
        vec![],
        None,
    );

    let output = x.max_pool((2, 2), (1, 1));
    output.backward();

    // MaxPool: градиент идёт только в те элементы, где был максимум
    // В каждом окне 2×2 максимум — правый нижний элемент
    // Для окна (0,0): максимум 5 (индекс 1,1) → градиент 1
    // Для окна (0,1): максимум 6 (индекс 1,2) → градиент 1
    // Для окна (1,0): максимум 8 (индекс 2,1) → градиент 1
    // Для окна (1,1): максимум 9 (индекс 2,2) → градиент 1
    let expected_grad = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];

    assert_eq!(x.tensor_data.borrow().grad, expected_grad);
}

// ===== 4. Производная свёртки + MaxPool =====
#[test]
fn test_conv2d_maxpool_backward() {
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
        true,
        vec![],
        None,
    );

    let kernel = Tensor::new(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1, 1, 2, 2],
        true,
        vec![],
        None,
    );

    let conv_out = x.conv2d(&kernel, (1, 1));
    let pool_out = conv_out.max_pool((2, 2), (1, 1));
    pool_out.backward();

    let grad_x = x.tensor_data.borrow().grad.clone();
    let grad_kernel = kernel.tensor_data.borrow().grad.clone();

    let expected_grad_x = vec![0., 0., 0., 0., 0., 1., 0., 2., 3.];
    let expected_grad_kernel = vec![5., 6., 8., 9.];

    assert_eq!(expected_grad_x, grad_x);
    assert_eq!(expected_grad_kernel, grad_kernel);
}

// ===== 5. Производная последовательных свёрток =====
#[test]
fn test_two_convs_backward() {
    let x = Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![1, 1, 3, 3],
        true,
        vec![],
        None,
    );

    let kernel1 = Tensor::new(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1, 1, 2, 2],
        true,
        vec![],
        None,
    );

    let kernel2 = Tensor::new(
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1, 1, 2, 2],
        true,
        vec![],
        None,
    );

    let out1 = x.conv2d(&kernel1, (1, 1));
    let out2 = out1.conv2d(&kernel2, (1, 1));
    out2.backward();

    let grad_x = x.tensor_data.borrow().grad.clone();
    let grad_k1 = kernel1.tensor_data.borrow().grad.clone();
    let grad_k2 = kernel2.tensor_data.borrow().grad.clone();

    let expected_grad_x = vec![0., 1., 0., 2., 3., 1., 0., 2., 3.];
    let expected_grad_k1 = vec![6., 8., 12., 14.];
    let expected_grad_k2 = vec![25., 31., 43., 49.];
}

#[cfg(feature = "slow")]
#[test]
fn test_syntetic_data() {
    let mut net = network::Network::new();
    let (w1, b1) = net.Linear(6, 10);
    let (w2, b2) = net.Linear(10, 1);

    let x: Vec<f32> = iproduct!([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
        .flat_map(|(a, b, c, d, e, f)| {
            vec![a as f32, b as f32, c as f32, d as f32, e as f32, f as f32]
        })
        .collect();

    let y: Vec<f32> = x
        .chunks(6)
        .filter_map(|chunk| chunk.get(2).copied())
        .collect();

    let x = Tensor::new(x.clone(), vec![x.len() / 6, 6], false, vec![], None);
    let y = Tensor::new(y.clone(), vec![y.len(), 1], false, vec![], None);

    let forward_fn = |x: &Tensor| x.matmul(&w1).add(&b1).tanh().matmul(&w2).add(&b2).tanh();

    let x_source = DataSource::Tensor(x.tensor_data.clone());
    let y_source = DataSource::Tensor(y.tensor_data.clone());

    net.fit(
        10000,
        0.001,
        x_source,
        y_source,
        Loss::MSE,
        150,
        0,
        forward_fn,
    );

    let output = forward_fn(&x);
    let final_loss = output.mse(&y);
    assert!(
        final_loss.tensor_data.borrow().data[0] <= 0.02,
        "MSE {} больше приемлемого значения 0.02",
        final_loss.tensor_data.borrow().data[0]
    );

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
}

#[cfg(feature = "slow")]
#[test]
fn test_iris() {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    fn load_iris_data(file_path: &str) -> (Tensor, Tensor) {
        let file = File::open(file_path).expect("Файл не найден");
        let reader = BufReader::new(file);

        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        for line in reader.lines() {
            let line = line.expect("Ошибка чтения строки");
            if line.starts_with("Id") || line.trim().is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 6 {
                continue;
            }

            let sepal_len: f32 = parts[1].parse().unwrap();
            let sepal_wid: f32 = parts[2].parse().unwrap();
            let petal_len: f32 = parts[3].parse().unwrap();
            let petal_wid: f32 = parts[4].parse().unwrap();

            x_data.push(sepal_len);
            x_data.push(sepal_wid);
            x_data.push(petal_len);
            x_data.push(petal_wid);

            let species = parts[5].trim().to_lowercase();
            let class_index = match species.as_str() {
                "iris-setosa" => 0,
                "iris-versicolor" => 1,
                "iris-virginica" => 2,
                _ => continue,
            };

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

            let mut pred_class = 0;
            let mut max_prob = pred_data.data[start];
            for j in 1..num_classes {
                let prob = pred_data.data[start + j];
                if prob > max_prob {
                    max_prob = prob;
                    pred_class = j;
                }
            }

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

    let x_source = DataSource::Tensor(x.tensor_data.clone());
    let y_source = DataSource::Tensor(y.tensor_data.clone());

    net.fit(
        5000,
        0.0001,
        x_source,
        y_source,
        Loss::CrossEntropyWithSoftmax,
        150,
        1000,
        forward_fn,
    );

    let output = net.forward(&x, forward_fn)._softmax();
    let acc = calculate_accuracy(&output, &y);
    assert!(acc > 0.95, "Ожидаемая точность: 0.95. Получено: {}", acc);
}
