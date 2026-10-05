use rayon::vec;

use crate::tensor::tensor::Operation;

use super::tensor::Tensor;
use std::rc::Rc;

#[test]
fn test_new_basic() {
    let t = Tensor::new(
        (0..12).map(|f| f as f32).collect(),
        vec![3, 4],
        true,
        vec![],
        None,
    );

    let td = t.tensor_data.borrow();
    assert_eq!(
        td.data,
        (0..12).map(|f| f as f32).collect::<Vec<f32>>(),
        "Данные при создании тензора не совпали"
    );
    assert_eq!(td.shape, vec![3, 4], "Размерность не совпала");
    // assert_eq!(td.stride, vec![2, 1]); Они есть, но ни разу не используются
    assert!(td.require_grad, "Неверный флаг взятия производной");
    assert!(
        td.parents.is_empty(),
        "Родителей при ручной создании быть не должно"
    );
    assert!(td.grad.is_empty(), "grad должен быть пустым при создании");
    assert!(td.operation.is_none(), "Руками был выставлен None");
}

#[test]
#[should_panic]
fn test_new_shape_mismatch() {
    // 5 элементов, но shape требует 6
    Tensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![2, 3],
        false,
        vec![],
        None,
    );
}

#[test]
fn test_uniform_in_range() {
    let t = Tensor::uniform(0.0, 1.0, vec![100], false);
    for v in t.tensor_data.borrow().data.iter() {
        assert!(*v >= 0.0 && *v <= 1.0, "value {} out of [0, 1]", v);
    }
}

#[test]
fn test_uniform_negative_range() {
    let t = Tensor::uniform(-1.0, 1.0, vec![100], false);
    for v in t.tensor_data.borrow().data.iter() {
        assert!(*v >= -1.0 && *v <= 1.0);
    }
}

#[test]
fn test_clone_is_rc_shallow() {
    // Clone Tensor — это клон Rc, не копия данных
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3], false, vec![], None);
    let b = a.clone();

    // Один и тот же Rc
    assert!(Rc::ptr_eq(&a.tensor_data, &b.tensor_data));

    // Изменение через b видно через a
    b.tensor_data.borrow_mut().data[0] = 99.0;
    assert_eq!(a.tensor_data.borrow().data[0], 99.0);
}

#[test]
fn test_matmul() {
    let a = Tensor::new(vec![1.0; 6], vec![2, 3], true, vec![], None);
    let b = Tensor::new(vec![2.0; 12], vec![3, 4], true, vec![], None);
    let c = a.matmul(&b);
    let expected_output_data = vec![6.0; 8];

    {
        let output = c.tensor_data.borrow();
        assert_eq!(
            output.shape,
            vec![2, 4],
            "Неверная размерность после матмула"
        );
        assert_eq!(
            output.parents.len(),
            2,
            "Неверное количество родителей у результата матмула"
        );
        assert!(
            Rc::ptr_eq(&output.parents[0], &a.tensor_data),
            "parents[0] должен быть a"
        );
        assert!(
            Rc::ptr_eq(&output.parents[1], &b.tensor_data),
            "parents[1] должен быть b"
        );
        assert!(matches!(output.operation, Some(Operation::Matmul)));
        assert!(output.require_grad);
        assert_eq!(
            output.data, expected_output_data,
            "Результат прямого матричного умножения неверен"
        );
    }

    c.backward();
    let a_grad = a.tensor_data.borrow().grad.clone();
    assert_eq!(a_grad, vec![8.0; 6], "Неверный градиент a");
    let b_grad = b.tensor_data.borrow().grad.clone();
    assert_eq!(b_grad, vec![2.0; 12], "Неверный градиент b");
}

#[test]
fn test_matmul_tall_wide() {
    let a = Tensor::new(
        vec![1., 2., 3., 4., 5., 6., 7., 8.],
        vec![4, 2],
        true,
        vec![],
        None,
    );
    let b = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3], true, vec![], None);
    let c = a.matmul(&b);

    {
        let output = c.tensor_data.borrow();
        assert_eq!(output.shape, vec![4, 3]);
        assert_eq!(
            output.data,
            vec![9., 12., 15., 19., 26., 33., 29., 40., 51., 39., 54., 69.]
        );
    }
    c.tensor_data.borrow_mut().grad = vec![1.; 12];
    c.backward();

    assert_eq!(
        a.tensor_data.borrow().grad,
        vec![6., 15., 6., 15., 6., 15., 6., 15.]
    );
    assert_eq!(
        b.tensor_data.borrow().grad,
        vec![16., 16., 16., 20., 20., 20.]
    );
}

#[test]
fn test_matmul_wide_tall() {
    let a = Tensor::new(
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
        vec![2, 5],
        true,
        vec![],
        None,
    );
    let b = Tensor::new(
        vec![1., 0., 0., 1., 1., 0., 0., 1., 1., 0.],
        vec![5, 2],
        true,
        vec![],
        None,
    );
    let c = a.matmul(&b);
    {
        let output = c.tensor_data.borrow();
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(output.data, vec![9., 6., 24., 16.]);
    }

    c.tensor_data.borrow_mut().grad = vec![1.; 4];
    c.backward();
    assert_eq!(a.tensor_data.borrow().grad, vec![1.; 10]);
    assert_eq!(
        b.tensor_data.borrow().grad,
        vec![7., 7., 9., 9., 11., 11., 13., 13., 15., 15.]
    );
}

#[test]
#[should_panic]
fn test_matmul_wrong_shape() {
    let a = Tensor::new(vec![1.; 6], vec![2, 3], true, vec![], None);
    let b = Tensor::new(vec![1.; 8], vec![2, 4], true, vec![], None);
    a.matmul(&b); // 3 != 2
}

// Блок с суммой
#[test]
fn test_add_same_shape() {
    let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3], true, vec![], None);
    let b = Tensor::new(
        vec![10., 20., 30., 40., 50., 60.],
        vec![2, 3],
        true,
        vec![],
        None,
    );
    let c = a.add(&b);
    {
        let output = c.tensor_data.borrow();
        assert_eq!(output.shape, vec![2, 3]);
        assert_eq!(output.data, vec![11., 22., 33., 44., 55., 66.]);
        assert_eq!(output.parents.len(), 2);
        assert!(Rc::ptr_eq(&output.parents[0], &a.tensor_data));
        assert!(Rc::ptr_eq(&output.parents[1], &b.tensor_data));
        assert!(matches!(output.operation, Some(Operation::Add)));
        assert!(output.require_grad);
    }
    c.tensor_data.borrow_mut().grad = vec![1.; 6];
    c.backward();
    assert_eq!(a.tensor_data.borrow().grad, vec![1.; 6]);
    assert_eq!(b.tensor_data.borrow().grad, vec![1.; 6]);
}

#[test]
fn test_add_broadcast_batch() {
    // [4,3] + [1,3]
    let a = Tensor::new(
        (0..12).map(|i| i as f32).collect(),
        vec![4, 3],
        true,
        vec![],
        None,
    );
    let b = Tensor::new(vec![10., 20., 30.], vec![1, 3], true, vec![], None);
    let c = a.add(&b);
    {
        let output = c.tensor_data.borrow();
        assert_eq!(output.shape, vec![4, 3]);
        let expected: Vec<f32> = (0..12).map(|i| i as f32 + [10., 20., 30.][i % 3]).collect();
        assert_eq!(output.data, expected);
    }
    c.tensor_data.borrow_mut().grad = vec![1.; 12];
    c.backward();
    assert_eq!(a.tensor_data.borrow().grad, vec![1.; 12]);
    assert_eq!(b.tensor_data.borrow().grad, vec![4., 4., 4.]);
}

// Relu
#[test]
fn test_relu() {
    let x = Tensor::new(
        vec![-2., -1., 0., 1., 2., -3.],
        vec![2, 3],
        true,
        vec![],
        None,
    );

    let y = x.relu();
    {
        let output = y.tensor_data.borrow();
        assert_eq!(output.shape, vec![2, 3]);
        assert_eq!(output.data, vec![0., 0., 0., 1., 2., 0.]);
        assert_eq!(output.parents.len(), 1);
        assert!(Rc::ptr_eq(&output.parents[0], &x.tensor_data));
        assert!(matches!(output.operation, Some(Operation::ReLu)));
        assert!(output.require_grad);
    }

    y.tensor_data.borrow_mut().grad = vec![1.; 6];
    y.backward();
    assert_eq!(x.tensor_data.borrow().grad, vec![0., 0., 0., 1., 1., 0.]);
}

#[test]
fn test_sigmoid() {
    let x = Tensor::new(vec![-1., 0., 1., 2.], vec![2, 2], true, vec![], None);
    let y = x.sigmoid();
    let data;
    {
        let output = y.tensor_data.borrow();
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(output.parents.len(), 1);
        assert!(matches!(output.operation, Some(Operation::Sigmoid)));
        data = output.data.clone();
    }
    let expected = [0.2689, 0.5, 0.7311, 0.8808];
    for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "sigmoid[{}]: {} != {}",
            i,
            got,
            exp
        );
    }
    y.tensor_data.borrow_mut().grad = vec![1.; 4];
    y.backward();
    let expected_grad: Vec<f32> = data.iter().map(|&s| s * (1.0 - s)).collect();
    let grad = x.tensor_data.borrow().grad.clone();
    for (i, (got, exp)) in grad.iter().zip(expected_grad.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-3, "grad[{}]: {} != {}", i, got, exp);
    }
}

#[test]
fn test_tanh_2d() {
    let x = Tensor::new(vec![-1., 0., 1., 2.], vec![2, 2], true, vec![], None);
    let y = x.tanh();
    let data;
    {
        let output = y.tensor_data.borrow();
        assert_eq!(output.shape, vec![2, 2]);
        assert!(matches!(output.operation, Some(Operation::Tanh)));
        data = output.data.clone();
    }
    let expected = [-0.7616, 0.0, 0.7616, 0.9640];
    for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-3, "tanh[{}]: {} != {}", i, got, exp);
    }
    y.tensor_data.borrow_mut().grad = vec![1.; 4];
    y.backward();
    let expected_grad: Vec<f32> = data.iter().map(|&t| 1.0 - t * t).collect();
    let grad = x.tensor_data.borrow().grad.clone();
    for (i, (got, exp)) in grad.iter().zip(expected_grad.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-3, "grad[{}]: {} != {}", i, got, exp);
    }
}

#[test]
fn test_pad_4d() {
    let x = Tensor::new(vec![1., 2., 3., 4.], vec![1, 1, 2, 2], true, vec![], None);
    let y = x.pad((1, 1, 1, 1));
    {
        let output = y.tensor_data.borrow();
        assert_eq!(output.shape, vec![1, 1, 4, 4]);
        assert_eq!(
            output.data,
            vec![
                0., 0., 0., 0., 0., 1., 2., 0., 0., 3., 4., 0., 0., 0., 0., 0.,
            ]
        );
        assert_eq!(output.parents.len(), 1);
        assert!(Rc::ptr_eq(&output.parents[0], &x.tensor_data));
        assert!(matches!(
            output.operation,
            Some(Operation::Padding(1, 1, 1, 1))
        ));
    }
    y.tensor_data.borrow_mut().grad = vec![1.; 16];
    y.backward();
    assert_eq!(x.tensor_data.borrow().grad, vec![1., 1., 1., 1.]);
}

#[test]
fn test_pad_batch() {
    // 2 картинки [2, 1, 2, 2]
    let x = Tensor::new(
        (0..8).map(|i| i as f32).collect(),
        vec![2, 1, 2, 2],
        true,
        vec![],
        None,
    );
    let y = x.pad((1, 1, 1, 1));
    {
        let output = y.tensor_data.borrow();
        assert_eq!(output.shape, vec![2, 1, 4, 4]);
    }
    y.tensor_data.borrow_mut().grad = vec![1.; 32];
    y.backward();
    assert_eq!(x.tensor_data.borrow().grad, vec![1.; 8]);
}

#[test]
fn test_reshape() {
    let x = Tensor::new(
        (0..6).map(|i| i as f32).collect(),
        vec![2, 3],
        true,
        vec![],
        None,
    );

    let y = x.reshape(vec![1, 6]);
    {
        let output = y.tensor_data.borrow();
        assert_eq!(output.shape, vec![1, 6]);
        assert_eq!(output.data, vec![0., 1., 2., 3., 4., 5.]);
        assert!(matches!(output.operation, Some(Operation::Reshape)));
    }

    y.tensor_data.borrow_mut().grad = vec![1.; 6];
    y.backward();
    assert_eq!(x.tensor_data.borrow().grad, vec![1.; 6]);
}

#[test]
fn test_flatten() {
    let x = Tensor::new(
        (0..120).map(|i| i as f32).collect(),
        vec![2, 3, 4, 5],
        true,
        vec![],
        None,
    );

    let y = x.flatten();
    {
        let output = y.tensor_data.borrow();
        assert_eq!(output.shape, vec![2, 60]);
        assert_eq!(output.data.len(), 120);
    }

    y.tensor_data.borrow_mut().grad = vec![1.; 120];
    y.backward();
    assert_eq!(x.tensor_data.borrow().grad, vec![1.; 120]);
}

#[test]
fn test_max_pool_basic() {
    let x = Tensor::new(
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
        vec![1, 1, 3, 3],
        true,
        vec![],
        None,
    );
    let y = x.max_pool((2, 2), (1, 1));
    {
        let output = y.tensor_data.borrow();
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
        assert_eq!(output.data, vec![5., 6., 8., 9.]);
        assert_eq!(output.parents.len(), 1);
        assert!(Rc::ptr_eq(&output.parents[0], &x.tensor_data));
        assert!(matches!(output.operation, Some(Operation::MaxPool(_, _))));
    }
    y.tensor_data.borrow_mut().grad = vec![1.; 4];
    y.backward();
    assert_eq!(
        x.tensor_data.borrow().grad,
        vec![0., 0., 0., 0., 1., 1., 0., 1., 1.]
    );
}

#[test]
fn test_max_pool_batch() {
    // [2, 1, 3, 3]
    let x = Tensor::new(
        (0..18).map(|i| i as f32).collect(),
        vec![2, 1, 3, 3],
        true,
        vec![],
        None,
    );
    let y = x.max_pool((2, 2), (1, 1));
    {
        let output = y.tensor_data.borrow();
        assert_eq!(output.shape, vec![2, 1, 2, 2]);
    }
    y.tensor_data.borrow_mut().grad = vec![1.; 8];
    y.backward();
    assert_eq!(x.tensor_data.borrow().grad.len(), 18);
    let sum: f32 = x.tensor_data.borrow().grad.iter().sum();
    assert_eq!(sum, 8.0);
}

#[test]
fn test_max_pool_2x2_stride_2() {
    let x = Tensor::new(
        (1..=16).map(|i| i as f32).collect(),
        vec![1, 1, 4, 4],
        true,
        vec![],
        None,
    );
    let y = x.max_pool((2, 2), (2, 2));
    {
        let output = y.tensor_data.borrow();
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
        // Окна: [1,2;5,6]→6, [3,4;7,8]→8, [9,10;13,14]→14, [11,12;15,16]→16
        assert_eq!(output.data, vec![6., 8., 14., 16.]);
        assert_eq!(output.parents.len(), 1);
        assert!(Rc::ptr_eq(&output.parents[0], &x.tensor_data));
        assert!(matches!(output.operation, Some(Operation::MaxPool(_, _))));
    }
    y.tensor_data.borrow_mut().grad = vec![1.; 4];
    y.backward();
    let grad = x.tensor_data.borrow().grad.clone();
    assert_eq!(grad.len(), 16);
    assert_eq!(
        grad,
        vec![
            0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1.,
        ],
        "Неверный градиент max_pool 2×2 stride 2"
    );
}

#[test]
fn test_max_pool_2x2_stride_3() {
    // stride=3, kernel=2 → окна НЕ перекрываются, часть пикселей вообще не входит
    let x = Tensor::new(
        (1..=25).map(|i| i as f32).collect(),
        vec![1, 1, 5, 5],
        true,
        vec![],
        None,
    );
    let y = x.max_pool((2, 2), (3, 3));
    {
        let output = y.tensor_data.borrow();
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
        assert_eq!(output.data, vec![7., 10., 22., 25.]);
    }
    y.tensor_data.borrow_mut().grad = vec![1.; 4];
    y.backward();
    let grad = x.tensor_data.borrow().grad.clone();
    assert_eq!(grad.len(), 25);
    let mut expected = vec![0.0_f32; 25];
    expected[6] = 1.0;
    expected[9] = 1.0;
    expected[21] = 1.0;
    expected[24] = 1.0;
    assert_eq!(grad, expected, "Неверный градиент max_pool 2×2 stride 3");
    // Проверка: пиксели вне окон получили 0
    assert_eq!(grad[2], 0.0, "пиксель (0,2) вне окон, но получил градиент");
    assert_eq!(grad[12], 0.0, "пиксель (2,2) вне окон, но получил градиент");
}

#[test]
fn test_max_pool_global() {
    // kernel = всё изображение
    let x = Tensor::new(
        (1..=16).map(|i| i as f32).collect(),
        vec![1, 1, 4, 4],
        true,
        vec![],
        None,
    );
    let y = x.max_pool((4, 4), (1, 1));
    {
        let output = y.tensor_data.borrow();
        assert_eq!(output.shape, vec![1, 1, 1, 1]);
        assert_eq!(output.data, vec![16.0]);
    }
    y.tensor_data.borrow_mut().grad = vec![1.0];
    y.backward();
    let grad = x.tensor_data.borrow().grad.clone();
    let mut expected = vec![0.0_f32; 16];
    expected[15] = 1.0;
    assert_eq!(grad, expected);
}

#[test]
fn test_max_pool_multi_channel() {
    // [1, 2, 4, 4] — 2 канала
    let x = Tensor::new(
        (0..32).map(|i| i as f32).collect(),
        vec![1, 2, 4, 4],
        true,
        vec![],
        None,
    );
    let y = x.max_pool((2, 2), (2, 2));
    {
        let output = y.tensor_data.borrow();
        assert_eq!(output.shape, vec![1, 2, 2, 2]);
    }
    y.tensor_data.borrow_mut().grad = vec![1.; 8];
    y.backward();
    // Каждый канал должен получить по 4 единицы
    let grad = x.tensor_data.borrow().grad.clone();
    assert_eq!(grad.len(), 32);
    let sum_channel0: f32 = grad[0..16].iter().sum();
    let sum_channel1: f32 = grad[16..32].iter().sum();
    assert_eq!(sum_channel0, 4.0, "канал 0: сумма градиентов != 4");
    assert_eq!(sum_channel1, 4.0, "канал 1: сумма градиентов != 4");
}

#[test]
fn test_max_pool_ties() {
    // [1, 1, 2, 2] = [[1, 1], [1, 1]] — все значения одинаковы
    let x = Tensor::new(vec![1., 1., 1., 1.], vec![1, 1, 2, 2], true, vec![], None);
    let y = x.max_pool((2, 2), (1, 1));
    {
        let output = y.tensor_data.borrow();
        assert_eq!(output.data, vec![1.0]);
    }
    y.tensor_data.borrow_mut().grad = vec![1.0];
    y.backward();
    let grad = x.tensor_data.borrow().grad.clone();
    // Получить градиент должен только первый элемент
    assert_eq!(grad, vec![1., 0., 0., 0.], "Неверный градиент");
    assert_eq!(grad.iter().sum::<f32>(), 1.0, "сумма != 1");
}

#[test]
fn test_conv2d_basic_gradients() {
    let x = Tensor::new(
        vec![0., 1., 2., 3., 4., 5., 6., 7., 8.],
        vec![1, 1, 3, 3],
        true,
        vec![],
        None,
    );
    let k = Tensor::new(vec![0., 1., 2., 3.], vec![1, 1, 2, 2], true, vec![], None);
    let out = x.conv2d(&k, (1, 1));
    {
        let output = out.tensor_data.borrow();
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
        assert_eq!(output.data, vec![19., 25., 37., 43.]);
        assert_eq!(output.parents.len(), 4);
        assert!(Rc::ptr_eq(&output.parents[2], &x.tensor_data));
        assert!(Rc::ptr_eq(&output.parents[3], &k.tensor_data));
    }
    out.tensor_data.borrow_mut().grad = vec![1.; 4];
    out.backward();
    let x_grad = x.tensor_data.borrow().grad.clone();
    assert_eq!(
        x_grad,
        vec![0., 1., 1., 2., 6., 4., 2., 5., 3.],
        "Неверный градиент по x"
    );
    let k_grad = k.tensor_data.borrow().grad.clone();
    assert_eq!(k_grad, vec![8., 12., 20., 24.], "Неверный градиент по ядру");
}

#[test]
fn test_conv2d_1x1_kernel() {
    let x = Tensor::new(vec![1., 2., 3., 4.], vec![1, 1, 2, 2], true, vec![], None);
    let k = Tensor::new(vec![2.0], vec![1, 1, 1, 1], true, vec![], None);
    let out = x.conv2d(&k, (1, 1));
    {
        let output = out.tensor_data.borrow();
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
        assert_eq!(output.data, vec![2., 4., 6., 8.]);
    }
    out.tensor_data.borrow_mut().grad = vec![1.; 4];
    out.backward();
    assert_eq!(x.tensor_data.borrow().grad, vec![2., 2., 2., 2.]);
    assert_eq!(k.tensor_data.borrow().grad, vec![10.]);
}

#[test]
fn test_conv2d_kernel_equals_input() {
    let x = Tensor::new(vec![1., 2., 3., 4.], vec![1, 1, 2, 2], true, vec![], None);
    let k = Tensor::new(vec![1., 1., 1., 1.], vec![1, 1, 2, 2], true, vec![], None);
    let out = x.conv2d(&k, (1, 1));
    {
        let output = out.tensor_data.borrow();
        assert_eq!(output.shape, vec![1, 1, 1, 1]);
        assert_eq!(output.data, vec![10.0]);
    }
    out.tensor_data.borrow_mut().grad = vec![1.0];
    out.backward();
    assert_eq!(x.tensor_data.borrow().grad, vec![1., 1., 1., 1.]);
    assert_eq!(k.tensor_data.borrow().grad, vec![1., 2., 3., 4.]);
}

#[test]
fn test_conv2d_stride_greater_than_kernel() {
    let x = Tensor::new(
        (0..25).map(|i| i as f32).collect(),
        vec![1, 1, 5, 5],
        true,
        vec![],
        None,
    );
    let k = Tensor::new(vec![1., 0., 0., 1.], vec![1, 1, 2, 2], true, vec![], None);
    let out = x.conv2d(&k, (3, 3));
    {
        let output = out.tensor_data.borrow();
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
    }
    out.tensor_data.borrow_mut().grad = vec![1.; 4];
    out.backward();
    let grad = x.tensor_data.borrow().grad.clone();
    assert_eq!(grad.len(), 25);
    assert_eq!(grad[2], 0.0, "пиксель (0,2) вне окон, grad != 0");
    assert_eq!(grad[10], 0.0, "пиксель (2,0) вне окон, grad != 0");
    assert_eq!(grad[12], 0.0, "пиксель (2,2) вне окон, grad != 0");
    assert_eq!(grad[22], 0.0, "пиксель (4,2) вне окон, grad != 0");
    assert_eq!(grad[0], 1.0);
    assert_eq!(grad[1], 0.0);
    assert_eq!(grad[5], 0.0);
    assert_eq!(grad[6], 1.0);
}

#[test]
fn test_conv2d_multi_channel() {
    let x = Tensor::new(
        vec![
            // канал 0
            0., 1., 2., 3., 4., 5., 6., 7., 8., // канал 1
            10., 11., 12., 13., 14., 15., 16., 17., 18.,
        ],
        vec![1, 2, 3, 3],
        true,
        vec![],
        None,
    );
    let k = Tensor::new(
        vec![1., 1., 1., 1., 1., 1., 1., 1.],
        vec![1, 2, 2, 2],
        true,
        vec![],
        None,
    );
    let out = x.conv2d(&k, (1, 1));
    {
        let output = out.tensor_data.borrow();
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
        assert_eq!(
            output.data,
            vec![56., 64., 80., 88.],
            "Неверный forward multi-channel"
        );
    }
    out.tensor_data.borrow_mut().grad = vec![1.; 4];
    out.backward();
    let x_grad = x.tensor_data.borrow().grad.clone();
    assert_eq!(
        x_grad,
        vec![
            1., 2., 1., 2., 4., 2., 1., 2., 1., // канал 0
            1., 2., 1., 2., 4., 2., 1., 2., 1., // канал 1
        ],
        "Неверный градиент по x (multi-channel)"
    );
    let k_grad = k.tensor_data.borrow().grad.clone();
    assert_eq!(
        k_grad,
        vec![8., 12., 20., 24., 48., 52., 60., 64.],
        "Неверный градиент по ядру (multi-channel)"
    );
}

#[test]
fn test_conv2d_batch() {
    let x = Tensor::new(
        (0..18).map(|i| i as f32).collect(),
        vec![2, 1, 3, 3],
        true,
        vec![],
        None,
    );
    let k = Tensor::new(vec![1., 0., 0., 1.], vec![1, 1, 2, 2], true, vec![], None);
    let out = x.conv2d(&k, (1, 1));
    {
        let output = out.tensor_data.borrow();
        assert_eq!(output.shape, vec![2, 1, 2, 2]);
        assert_eq!(
            output.data,
            vec![4., 6., 10., 12., 22., 24., 28., 30.],
            "Неверный forward batch"
        );
    }
    out.tensor_data.borrow_mut().grad = vec![1.; 8];
    out.backward();
    let x_grad = x.tensor_data.borrow().grad.clone();
    assert_eq!(
        x_grad,
        vec![
            1., 1., 0., 1., 2., 1., 0., 1., 1., // батч 0
            1., 1., 0., 1., 2., 1., 0., 1., 1., // батч 1
        ],
        "Неверный градиент по x (batch)"
    );
    let k_grad = k.tensor_data.borrow().grad.clone();
    assert_eq!(
        k_grad,
        vec![52., 60., 76., 84.],
        "Неверный градиент по ядру (batch)"
    );
}

#[test]
fn test_conv2d_batch_multi_channel() {
    let x = Tensor::new(
        vec![
            // батч 0, канал 0
            1., 2., 3., 4., 5., 6., 7., 8., 9., // батч 0, канал 1
            10., 11., 12., 13., 14., 15., 16., 17., 18., // батч 1, канал 0
            19., 20., 21., 22., 23., 24., 25., 26., 27., // батч 1, канал 1
            28., 29., 30., 31., 32., 33., 34., 35., 36.,
        ],
        vec![2, 2, 3, 3],
        true,
        vec![],
        None,
    );
    let k = Tensor::new(
        vec![1., 1., 1., 1., 1., 1., 1., 1.],
        vec![1, 2, 2, 2],
        true,
        vec![],
        None,
    );
    let out = x.conv2d(&k, (1, 1));
    {
        let output = out.tensor_data.borrow();
        assert_eq!(output.shape, vec![2, 1, 2, 2]);
        assert_eq!(
            output.data,
            vec![60., 68., 84., 92., 204., 212., 228., 236.],
            "Неверный forward batch+multi-channel"
        );
    }
    out.tensor_data.borrow_mut().grad = vec![1.; 8];
    out.backward();
    let x_grad = x.tensor_data.borrow().grad.clone();
    let pattern = vec![1., 2., 1., 2., 4., 2., 1., 2., 1.];
    let mut expected_x_grad = vec![];
    for _ in 0..4 {
        expected_x_grad.extend_from_slice(&pattern);
    }
    assert_eq!(x_grad, expected_x_grad, "Неверный градиент по x");

    let k_grad = k.tensor_data.borrow().grad.clone();
    assert_eq!(
        k_grad,
        vec![96., 104., 120., 128., 168., 176., 192., 200.],
        "Неверный градиент по ядру"
    );
}

#[test]
fn test_conv2d_asymmetric_stride() {
    let x = Tensor::new(
        (0..25).map(|i| i as f32).collect(),
        vec![1, 1, 5, 5],
        true,
        vec![],
        None,
    );
    let k = Tensor::new(vec![1., 1., 1., 1.], vec![1, 1, 2, 2], true, vec![], None);
    let out = x.conv2d(&k, (2, 1));
    {
        let output = out.tensor_data.borrow();
        assert_eq!(output.shape, vec![1, 1, 2, 4]);
        assert_eq!(
            output.data,
            vec![12., 16., 20., 24., 52., 56., 60., 64.],
            "Неверный forward asymmetric stride"
        );
    }
    out.tensor_data.borrow_mut().grad = vec![1.; 8];
    out.backward();
    let x_grad = x.tensor_data.borrow().grad.clone();
    assert_eq!(
        x_grad,
        vec![
            1., 2., 2., 2., 1., 1., 2., 2., 2., 1., 1., 2., 2., 2., 1., 1., 2., 2., 2., 1., 0., 0.,
            0., 0., 0.,
        ],
        "Неверный градиент по x (asymmetric stride)"
    );
    let k_grad = k.tensor_data.borrow().grad.clone();
    assert_eq!(
        k_grad,
        vec![52., 60., 92., 100.],
        "Неверный градиент по ядру (asymmetric stride)"
    );
}

#[test]
fn test_mse() {
    let pred = Tensor::new(vec![1., 2., 3., 4.], vec![1, 4], true, vec![], None);
    let target = Tensor::new(vec![2., 2., 2., 2.], vec![1, 4], false, vec![], None);
    let loss = pred.mse(&target);
    {
        let output = loss.tensor_data.borrow();
        assert_eq!(output.shape, vec![1]);
        assert!((output.data[0] - 1.5).abs() < 1e-6);
        assert!(matches!(output.operation, Some(Operation::MSE)));
    }
    loss.backward();
    let grad = pred.tensor_data.borrow().grad.clone();
    assert!((grad[0] - (-0.5)).abs() < 1e-6);
    assert!((grad[1] - 0.0).abs() < 1e-6);
    assert!((grad[2] - 0.5).abs() < 1e-6);
    assert!((grad[3] - 1.0).abs() < 1e-6);
}

#[test]
fn test_cross_entropy() {
    let logits = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2], true, vec![], None);
    let target = Tensor::new(vec![1., 0., 0., 1.], vec![2, 2], false, vec![], None);
    let loss = logits.cross_entropy_with_softmax(&target);
    {
        let output = loss.tensor_data.borrow();
        assert!(
            (output.data[0] - 0.813).abs() < 1e-2,
            "loss = {}",
            output.data[0]
        );
    }
    loss.backward();
    assert_eq!(logits.tensor_data.borrow().grad.len(), 4);
}

#[test]
fn test_cross_entropy_stability() {
    // Экстремальные логиты — softmax должен быть стабильным
    let logits = Tensor::new(
        vec![1000., 1000., 1000., 1000.],
        vec![1, 4],
        true,
        vec![],
        None,
    );
    let target = Tensor::new(vec![1., 0., 0., 0.], vec![1, 4], false, vec![], None);
    let loss = logits.cross_entropy_with_softmax(&target);
    let v = loss.tensor_data.borrow().data[0];
    // Все логиты равны, softmax = [0.25;4], loss = -log(0.25) = 1.386
    assert!((v - 1.386).abs() < 1e-3, "нестабильный softmax: {}", v);
    assert!(v.is_finite(), "loss = {} (NaN/inf)", v);
}

// ==================== HE-UNIFORM ====================

#[test]
fn test_he_uniform_bound_linear() {
    // Linear [100, 50]: fan_in = 100, bound = sqrt(6/100) ≈ 0.2449
    let t = Tensor::he_uniform(vec![100, 50], false);
    let data = &t.tensor_data.borrow().data;
    let expected_bound = (6.0_f32 / 100.0).sqrt();

    for v in data {
        assert!(
            v.abs() <= expected_bound + 1e-6,
            "v = {} вне [-{}, {}]",
            v,
            expected_bound,
            expected_bound
        );
    }
}

#[test]
fn test_he_uniform_bound_conv() {
    // Conv [32, 3, 3, 3]: fan_in = 3*3*3 = 27, bound = sqrt(6/27) ≈ 0.4714
    let t = Tensor::he_uniform(vec![32, 3, 3, 3], false);
    let data = &t.tensor_data.borrow().data;
    let expected_bound = (6.0_f32 / 27.0).sqrt();

    for v in data {
        assert!(v.abs() <= expected_bound + 1e-6);
    }
}

#[test]
fn test_he_uniform_variance() {
    // Для большого тензора var ≈ bound² / 3 = 2 / fan_in
    let t = Tensor::he_uniform(vec![1000, 100], false);
    let data = &t.tensor_data.borrow().data;

    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let var: f32 = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;

    // Var ≈ 2 / fan_in = 2 / 1000 = 0.002
    assert!(
        (var - 0.002).abs() < 0.0005,
        "var = {} (ожидалось 0.002)",
        var
    );
}

#[test]
#[should_panic]
fn test_he_uniform_wrong_ndim() {
    // 3D не поддерживается
    Tensor::he_uniform(vec![2, 3, 4], false);
}
