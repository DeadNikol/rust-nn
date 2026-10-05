use super::network::{Loss, Network};
use crate::{
    batch_iterator::batch_iterator::DataSource, network::network::accuracy, tensor::tensor::Tensor,
};
use std::rc::Rc;

#[test]
fn test_network_new_empty() {
    let net = Network::new();
    assert!(net.parametres.is_empty());
}

#[test]
fn test_linear_adds_two_params() {
    let mut net = Network::new();
    assert_eq!(net.parametres.len(), 0);

    let (_w, _b) = net.Linear(4, 3);
    assert_eq!(net.parametres.len(), 2, "Linear должен добавить w и b");
}
#[test]
fn test_linear_params_same_rc() {
    let mut net = Network::new();
    let (w, b) = net.Linear(4, 3);

    assert!(
        Rc::ptr_eq(&net.parametres[0], &w.tensor_data),
        "parametres[0] должен быть тот же Rc, что w"
    );
    assert!(
        Rc::ptr_eq(&net.parametres[1], &b.tensor_data),
        "parametres[1] должен быть тот же Rc, что b"
    );
}

#[test]
fn test_conv2d_adds_one_param() {
    let mut net = Network::new();
    let _k = net.Conv2d(1, 32, (3, 3));
    assert_eq!(net.parametres.len(), 1, "Conv2d должен добавить ядро");
}
#[test]
fn test_conv2d_param_same_rc() {
    let mut net = Network::new();
    let k = net.Conv2d(1, 32, (3, 3));

    assert!(
        Rc::ptr_eq(&net.parametres[0], &k.tensor_data),
        "parametres[0] должен быть тот же Rc, что ядро"
    );
}

#[test]
fn test_multiple_layers_order() {
    let mut net = Network::new();
    let (w1, b1) = net.Linear(4, 3);
    let k = net.Conv2d(1, 8, (3, 3));
    let (w2, b2) = net.Linear(2, 1);

    // Порядок: w1, b1, k, w2, b2
    assert_eq!(net.parametres.len(), 5);
    assert!(Rc::ptr_eq(&net.parametres[0], &w1.tensor_data));
    assert!(Rc::ptr_eq(&net.parametres[1], &b1.tensor_data));
    assert!(Rc::ptr_eq(&net.parametres[2], &k.tensor_data));
    assert!(Rc::ptr_eq(&net.parametres[3], &w2.tensor_data));
    assert!(Rc::ptr_eq(&net.parametres[4], &b2.tensor_data));
}

#[test]
fn test_linear_shapes() {
    let mut net = Network::new();
    let (w, b) = net.Linear(4, 3);

    assert_eq!(w.tensor_data.borrow().shape, vec![4, 3]);
    assert_eq!(b.tensor_data.borrow().shape, vec![1, 3]);
    assert!(w.tensor_data.borrow().require_grad);
    assert!(b.tensor_data.borrow().require_grad);
}

#[test]
fn test_conv2d_shapes() {
    let mut net = Network::new();
    let k = net.Conv2d(3, 8, (5, 5));

    // [out_channels, in_channels, kH, kW]
    assert_eq!(k.tensor_data.borrow().shape, vec![8, 3, 5, 5]);
    assert!(k.tensor_data.borrow().require_grad);
}

#[test]
fn test_forward_applies_fn() {
    let net = Network::new();
    let x = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2], false, vec![], None);

    let out = net.forward(&x, |x| x.relu());
    assert_eq!(out.tensor_data.borrow().data, vec![1., 2., 3., 4.]);
}

#[test]
fn test_forward_does_not_mutate_input() {
    let net = Network::new();
    let x = Tensor::new(vec![-1., 2., -3., 4.], vec![2, 2], false, vec![], None);

    let _ = net.forward(&x, |x| x.relu());
    // x не должен измениться
    assert_eq!(x.tensor_data.borrow().data, vec![-1., 2., -3., 4.]);
}

#[test]
fn test_forward_real_network_shape() {
    let mut net = Network::new();
    let (w, b) = net.Linear(4, 3);

    let x = Tensor::new(vec![1., 2., 3., 4.], vec![1, 4], false, vec![], None);
    let out = net.forward(&x, |x| x.matmul(&w).add(&b));

    assert_eq!(out.tensor_data.borrow().shape, vec![1, 3]);
}

#[test]
fn test_fit_linear_regression() {
    let net = Network::new();

    // W: [1, 1], b: [1, 1]
    let mut net = Network::new();
    let (w, b) = net.Linear(1, 1);

    // Данные: y = 2x + 1
    let x = Tensor::new(vec![1., 2., 3., 4., 5.], vec![5, 1], false, vec![], None);
    let y = Tensor::new(vec![3., 5., 7., 9., 11.], vec![5, 1], false, vec![], None);

    let x_source = DataSource::Tensor(x.tensor_data.clone());
    let y_source = DataSource::Tensor(y.tensor_data.clone());

    let forward_fn = |x: &Tensor| x.matmul(&w).add(&b);

    net.fit(
        200,  // epochs
        0.05, // lr
        x_source,
        y_source,
        Loss::MSE,
        5, // batch_size = все данные
        0, // verbose = 0 (без вывода)
        forward_fn,
    );

    // Проверим, что w ≈ 2, b ≈ 1
    let w_val = w.tensor_data.borrow().data[0];
    let b_val = b.tensor_data.borrow().data[0];
    assert!((w_val - 2.0).abs() < 0.1, "w = {} (ожидалось 2.0)", w_val);
    assert!((b_val - 1.0).abs() < 0.1, "b = {} (ожидалось 1.0)", b_val);

    // Проверим, что loss мал
    let pred = forward_fn(&x);
    let final_loss = pred.mse(&y).tensor_data.borrow().data[0];
    assert!(
        final_loss < 0.01,
        "final loss = {} (ожидалось < 0.01)",
        final_loss
    );
}

#[test]
fn test_fit_logistic_regression() {
    let mut net = Network::new();
    let (w, b) = net.Linear(1, 2);

    let x = Tensor::new(vec![-2., -1., 1., 2.], vec![4, 1], false, vec![], None);
    let y = Tensor::new(
        vec![
            1., 0., // x = -2 → класс 0
            1., 0., // x = -1 → класс 0
            0., 1., // x =  1 → класс 1
            0., 1., // x =  2 → класс 1
        ],
        vec![4, 2],
        false,
        vec![],
        None,
    );

    let x_source = DataSource::Tensor(x.tensor_data.clone());
    let y_source = DataSource::Tensor(y.tensor_data.clone());

    let forward_fn = |x: &Tensor| x.matmul(&w).add(&b);

    net.fit(
        300,
        0.1,
        x_source,
        y_source,
        Loss::CrossEntropyWithSoftmax,
        4,
        0,
        forward_fn,
    );

    // Проверим, что все примеры классифицированы правильно
    let pred = forward_fn(&x);
    let pred_data = pred.tensor_data.borrow().data.clone();
    for i in 0..4 {
        let p0 = pred_data[i * 2];
        let p1 = pred_data[i * 2 + 1];
        let target0 = y.tensor_data.borrow().data[i * 2];

        if target0 > 0.5 {
            assert!(
                p0 > p1,
                "пример {}: ожидался класс 0, но p0={} p1={}",
                i,
                p0,
                p1
            );
        } else {
            assert!(
                p1 > p0,
                "пример {}: ожидался класс 1, но p0={} p1={}",
                i,
                p0,
                p1
            );
        }
    }
}

#[test]
fn test_fit_xor() {
    let mut net = Network::new();
    let (w1, b1) = net.Linear(2, 4);
    let (w2, b2) = net.Linear(4, 2);

    let x = Tensor::new(
        vec![0., 0., 0., 1., 1., 0., 1., 1.],
        vec![4, 2],
        false,
        vec![],
        None,
    );
    let y = Tensor::new(
        vec![
            1., 0., // 0 XOR 0 = 0
            0., 1., // 0 XOR 1 = 1
            0., 1., // 1 XOR 0 = 1
            1., 0., // 1 XOR 1 = 0
        ],
        vec![4, 2],
        false,
        vec![],
        None,
    );

    let x_source = DataSource::Tensor(x.tensor_data.clone());
    let y_source = DataSource::Tensor(y.tensor_data.clone());

    let forward_fn = |x: &Tensor| x.matmul(&w1).add(&b1).relu().matmul(&w2).add(&b2);

    net.fit(
        1000,
        0.5,
        x_source,
        y_source,
        Loss::CrossEntropyWithSoftmax,
        4,
        0,
        forward_fn,
    );

    // Проверим классификацию
    let pred = forward_fn(&x);
    let pred_data = pred.tensor_data.borrow().data.clone();
    let mut correct = 0;
    for i in 0..4 {
        let p0 = pred_data[i * 2];
        let p1 = pred_data[i * 2 + 1];
        let pred_class = if p0 > p1 { 0 } else { 1 };
        let target_class = if y.tensor_data.borrow().data[i * 2] > 0.5 {
            0
        } else {
            1
        };
        if pred_class == target_class {
            correct += 1;
        }
    }
    assert_eq!(correct, 4, "XOR: {} / 4 правильно", correct);
    assert!(
        correct >= 3,
        "XOR: {} / 4 правильно (ожидалось хотя бы 3)",
        correct
    );
}

#[test]
fn test_fit_xor_light() {
    let mut net = Network::new();
    let (w1, b1) = net.Linear(2, 4);
    let (w2, b2) = net.Linear(4, 2);

    let x = Tensor::new(vec![0., 0., 0., 1., 1., 0., 1., 1.], vec![4, 2], false, vec![], None);
    let y = Tensor::new(vec![1., 0., 0., 1., 0., 1., 1., 0.], vec![4, 2], false, vec![], None);

    let x_source = DataSource::Tensor(x.tensor_data.clone());
    let y_source = DataSource::Tensor(y.tensor_data.clone());

    let forward_fn = |x: &Tensor| {
        x.matmul(&w1).add(&b1).relu().matmul(&w2).add(&b2)
    };

    net.fit(2000, 0.5, x_source, y_source, Loss::CrossEntropyWithSoftmax, 4, 0, forward_fn);

    let pred = forward_fn(&x);
    let pred_data = pred.tensor_data.borrow().data.clone();
    let mut correct = 0;
    for i in 0..4 {
        let p0 = pred_data[i * 2];
        let p1 = pred_data[i * 2 + 1];
        let pred_class = if p0 > p1 { 0 } else { 1 };
        let target_class = if y.tensor_data.borrow().data[i * 2] > 0.5 { 0 } else { 1 };
        if pred_class == target_class { correct += 1; }
    }
    assert!(correct >= 3, "XOR: {} / 4 (ожидалось хотя бы 3)", correct);
}

#[test]
fn test_accuracy_all_correct() {
    let pred = Tensor::new(
        vec![0.9, 0.1, 0.1, 0.9], // argmax: [0, 1]
        vec![2, 2],
        false,
        vec![],
        None,
    );
    let target = Tensor::new(
        vec![1., 0., 0., 1.], // argmax: [0, 1]
        vec![2, 2],
        false,
        vec![],
        None,
    );
    assert_eq!(accuracy(&pred, &target), 1.0);
}

#[test]
fn test_accuracy_all_wrong() {
    let pred = Tensor::new(
        vec![0.1, 0.9, 0.9, 0.1], // argmax: [1, 0]
        vec![2, 2],
        false,
        vec![],
        None,
    );
    let target = Tensor::new(
        vec![1., 0., 0., 1.], // argmax: [0, 1]
        vec![2, 2],
        false,
        vec![],
        None,
    );
    assert_eq!(accuracy(&pred, &target), 0.0);
}

#[test]
fn test_accuracy_half() {
    let pred = Tensor::new(
        vec![0.9, 0.1, 0.9, 0.1], // argmax: [0, 0]
        vec![2, 2],
        false,
        vec![],
        None,
    );
    let target = Tensor::new(
        vec![1., 0., 0., 1.], // argmax: [0, 1]
        vec![2, 2],
        false,
        vec![],
        None,
    );
    assert_eq!(accuracy(&pred, &target), 0.5);
}

#[test]
fn test_fit_two_epochs_no_grad_accumulation() {
    let mut net = Network::new();
    let (w, b) = net.Linear(1, 1); // ← используем и b

    let x = Tensor::new(vec![1., 2., 3.], vec![3, 1], false, vec![], None);
    let y = Tensor::new(vec![1., 2., 3.], vec![3, 1], false, vec![], None);

    let x_source = DataSource::Tensor(x.tensor_data.clone());
    let y_source = DataSource::Tensor(y.tensor_data.clone());

    let forward_fn = |x: &Tensor| x.matmul(&w).add(&b); // ← используем b

    net.fit(
        1,
        0.1,
        x_source.clone(),
        y_source.clone(),
        Loss::MSE,
        3,
        0,
        forward_fn,
    );
    let w_after_epoch1 = w.tensor_data.borrow().data[0];

    net.fit(1, 0.1, x_source, y_source, Loss::MSE, 3, 0, forward_fn);
    let w_after_epoch2 = w.tensor_data.borrow().data[0];

    assert_ne!(
        w_after_epoch1, w_after_epoch2,
        "веса не меняются между эпохами"
    );
}
#[test]
fn test_conv_network_shape() {
    let mut net = Network::new();
    let conv = net.Conv2d(1, 8, (3, 3));
    let (w, b) = net.Linear(8, 10); // ← 8, не 32

    let x = Tensor::new(
        (0..16).map(|i| i as f32 / 10.).collect(),
        vec![1, 1, 4, 4],
        false,
        vec![],
        None,
    );

    let forward_fn = |x: &Tensor| {
        x.conv2d(&conv, (1, 1)) // [1,8,2,2]
            .max_pool((2, 2), (2, 2)) // [1,8,1,1]
            .relu()
            .flatten() // [1,8]
            .matmul(&w) // [1,8] @ [8,10] = [1,10]
            .add(&b)
    };

    let out = net.forward(&x, forward_fn);
    assert_eq!(out.tensor_data.borrow().shape, vec![1, 10]);
}
