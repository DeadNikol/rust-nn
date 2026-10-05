use super::optims::SGD;
use crate::tensor::tensor::Tensor;
use std::rc::Rc;

#[test]
fn test_sgd_velocity_size() {
    let w1 = Tensor::new(vec![0.0; 6], vec![2, 3], true, vec![], None);
    let w2 = Tensor::new(vec![0.0; 4], vec![2, 2], true, vec![], None);

    let sgd = SGD::new(
        vec![w1.tensor_data.clone(), w2.tensor_data.clone()],
        0.1,
        0.0,
    );

    assert_eq!(sgd.velosity.len(), 2, "должно быть 2 velocity-вектора");
    assert_eq!(sgd.velosity[0].len(), 6, "velocity[0] должен быть длиной 6");
    assert_eq!(sgd.velosity[1].len(), 4, "velocity[1] должен быть длиной 4");
}

#[test]
fn test_sgd_velocity_initial_zero() {
    let w = Tensor::new(vec![1.0; 3], vec![1, 3], true, vec![], None);
    let sgd = SGD::new(vec![w.tensor_data.clone()], 0.1, 0.9);

    for v in &sgd.velosity[0] {
        assert_eq!(*v, 0.0, "velocity должен быть 0 при создании");
    }
}

#[test]
fn test_sgd_step_no_momentum_exact() {
    let w = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3], true, vec![], None);
    w.tensor_data.borrow_mut().grad = vec![0.5, 0.25, 0.125];

    let mut sgd = SGD::new(vec![w.tensor_data.clone()], 0.5, 0.0);
    sgd.backward_step();

    assert_eq!(
        w.tensor_data.borrow().data,
        vec![0.75, 1.875, 2.9375],
        "неверное обновление без momentum"
    );
}

#[test]
fn test_sgd_multiple_steps_no_momentum_exact() {
    let w = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3], true, vec![], None);
    let mut sgd = SGD::new(vec![w.tensor_data.clone()], 0.5, 0.0);

    for _ in 0..3 {
        w.tensor_data.borrow_mut().grad = vec![0.5, 0.25, 0.125];
        sgd.backward_step();
    }

    assert_eq!(w.tensor_data.borrow().data, vec![0.25, 1.625, 2.8125]);
}

#[test]
fn test_sgd_multiple_params_exact() {
    let w1 = Tensor::new(vec![1.0, 2.0], vec![1, 2], true, vec![], None);
    let w2 = Tensor::new(vec![4.0, 8.0, 16.0], vec![1, 3], true, vec![], None);

    w1.tensor_data.borrow_mut().grad = vec![0.5, 0.25];
    w2.tensor_data.borrow_mut().grad = vec![0.125, 0.0625, 0.03125];

    let mut sgd = SGD::new(
        vec![w1.tensor_data.clone(), w2.tensor_data.clone()],
        0.5,
        0.0,
    );
    sgd.backward_step();

    assert_eq!(w1.tensor_data.borrow().data, vec![0.75, 1.875]);
    assert_eq!(
        w2.tensor_data.borrow().data,
        vec![3.9375, 7.96875, 15.984375]
    );
}

#[test]
fn test_sgd_lr_one_exact() {
    let w = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3], true, vec![], None);
    w.tensor_data.borrow_mut().grad = vec![0.5, 0.25, 0.125];

    let mut sgd = SGD::new(vec![w.tensor_data.clone()], 1.0, 0.0);
    sgd.backward_step();

    assert_eq!(w.tensor_data.borrow().data, vec![0.5, 1.75, 2.875]);
}

#[test]
fn test_sgd_momentum_first_step_exact() {
    let w = Tensor::new(vec![1.0], vec![1, 1], true, vec![], None);
    w.tensor_data.borrow_mut().grad = vec![1.0];

    let mut sgd = SGD::new(vec![w.tensor_data.clone()], 0.5, 0.5);
    sgd.backward_step();

    assert_eq!(w.tensor_data.borrow().data, vec![0.5]);
    assert_eq!(sgd.velosity[0], vec![0.5]);
}

#[test]
fn test_sgd_momentum_accumulates_exact() {
    let w = Tensor::new(vec![1.0], vec![1, 1], true, vec![], None);
    let mut sgd = SGD::new(vec![w.tensor_data.clone()], 0.5, 0.5);

    // Шаг 1
    w.tensor_data.borrow_mut().grad = vec![1.0];
    sgd.backward_step();
    assert_eq!(w.tensor_data.borrow().data, vec![0.5], "шаг 1: w");
    assert_eq!(sgd.velosity[0], vec![0.5], "шаг 1: v");

    // Шаг 2
    w.tensor_data.borrow_mut().grad = vec![1.0];
    sgd.backward_step();
    assert_eq!(w.tensor_data.borrow().data, vec![-0.25], "шаг 2: w");
    assert_eq!(sgd.velosity[0], vec![0.75], "шаг 2: v");

    // Шаг 3
    w.tensor_data.borrow_mut().grad = vec![1.0];
    sgd.backward_step();
    assert_eq!(w.tensor_data.borrow().data, vec![-1.125], "шаг 3: w");
    assert_eq!(sgd.velosity[0], vec![0.875], "шаг 3: v");
}

#[test]
fn test_sgd_momentum_persists_exact() {
    let w = Tensor::new(vec![1.0], vec![1, 1], true, vec![], None);
    let mut sgd = SGD::new(vec![w.tensor_data.clone()], 0.5, 0.5);

    // Шаг 1: grad = 1
    w.tensor_data.borrow_mut().grad = vec![1.0];
    sgd.backward_step();

    // Шаг 2: grad = 0
    w.tensor_data.borrow_mut().grad = vec![0.0];
    sgd.backward_step();

    assert_eq!(sgd.velosity[0], vec![0.25], "v сохраняется");
    assert_eq!(
        w.tensor_data.borrow().data,
        vec![0.25],
        "w продолжил двигаться"
    );
}

#[test]
fn test_sgd_momentum_formula_exact() {
    let w = Tensor::new(vec![0.0], vec![1, 1], true, vec![], None);
    let mut sgd = SGD::new(vec![w.tensor_data.clone()], 0.5, 0.5);

    // Шаг 1
    w.tensor_data.borrow_mut().grad = vec![1.0];
    sgd.backward_step();
    assert_eq!(sgd.velosity[0], vec![0.5]);
    assert_eq!(w.tensor_data.borrow().data, vec![-0.5]);

    // Шаг 2
    w.tensor_data.borrow_mut().grad = vec![2.0];
    sgd.backward_step();
    assert_eq!(sgd.velosity[0], vec![1.25]);
    assert_eq!(w.tensor_data.borrow().data, vec![-1.75]);

    // Шаг 3
    w.tensor_data.borrow_mut().grad = vec![4.0];
    sgd.backward_step();
    assert_eq!(sgd.velosity[0], vec![2.625]);
    assert_eq!(w.tensor_data.borrow().data, vec![-4.375]);
}

#[test]
fn test_sgd_zero_grad_exact() {
    let w = Tensor::new(vec![1.0; 3], vec![1, 3], true, vec![], None);
    w.tensor_data.borrow_mut().grad = vec![0.5, 0.25, 0.125];

    let mut sgd = SGD::new(vec![w.tensor_data.clone()], 0.5, 0.0);
    sgd.backward_step();

    assert_eq!(w.tensor_data.borrow().grad, vec![0.0, 0.0, 0.0]);
    assert_eq!(w.tensor_data.borrow().grad.len(), 3);
}

#[test]
fn test_sgd_zero_grad_visible_through_other_rc_exact() {
    let w = Tensor::new(vec![1.0; 3], vec![1, 3], true, vec![], None);
    let w_rc = w.tensor_data.clone();

    assert!(
        Rc::ptr_eq(&w.tensor_data, &w_rc),
        "Rc должны быть одним и тем же"
    );

    w.tensor_data.borrow_mut().grad = vec![0.5, 0.25, 0.125];

    let mut sgd = SGD::new(vec![w_rc.clone()], 0.5, 0.0);
    assert_eq!(sgd.parametres[0].borrow().grad, vec![0.5, 0.25, 0.125]);

    sgd.backward_step();

    assert_eq!(sgd.parametres[0].borrow().grad, vec![0.0, 0.0, 0.0]);
    assert_eq!(
        w.tensor_data.borrow().grad,
        vec![0.0, 0.0, 0.0],
        "grad не обнулён в Tensor-обёртке!"
    );
}

#[test]
fn test_sgd_after_real_backward_exact() {
    let w = Tensor::new(vec![1.0, 1.0], vec![2, 1], true, vec![], None);
    let x = Tensor::new(vec![1.0, 1.0], vec![1, 2], true, vec![], None);
    let target = Tensor::new(vec![0.0], vec![1, 1], false, vec![], None);

    let mut sgd = SGD::new(vec![w.tensor_data.clone()], 0.5, 0.0);

    let pred = x.matmul(&w); // [2.0]
    let loss = pred.mse(&target); // (2-0)² = 4
    loss.backward();
    assert_eq!(
        w.tensor_data.borrow().grad,
        vec![4.0, 4.0],
        "grad после backward"
    );

    sgd.backward_step();
    assert_eq!(
        w.tensor_data.borrow().data,
        vec![-1.0, -1.0],
        "w после step"
    );
    assert_eq!(
        w.tensor_data.borrow().grad,
        vec![0.0, 0.0],
        "grad после step"
    );

    let pred2 = x.matmul(&w); // [-2.0]
    let loss2 = pred2.mse(&target); // (−2−0)² = 4
    loss2.backward();
    assert_eq!(
        w.tensor_data.borrow().grad,
        vec![-4.0, -4.0],
        "grad после 2-го backward"
    );
}

#[test]
fn test_sgd_lr_zero() {
    let w = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3], true, vec![], None);
    w.tensor_data.borrow_mut().grad = vec![1.0, 1.0, 1.0];

    let mut sgd = SGD::new(vec![w.tensor_data.clone()], 0.0, 0.5);
    sgd.backward_step();

    assert_eq!(w.tensor_data.borrow().data, vec![1.0, 2.0, 3.0]);
    assert_eq!(sgd.velosity[0], vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_sgd_zero_grad_first() {
    let w = Tensor::new(vec![1.0], vec![1, 1], true, vec![], None);
    w.tensor_data.borrow_mut().grad = vec![0.0];

    let mut sgd = SGD::new(vec![w.tensor_data.clone()], 0.5, 0.5);
    sgd.backward_step();

    // v = 0, w = 1
    assert_eq!(w.tensor_data.borrow().data, vec![1.0]);
    assert_eq!(sgd.velosity[0], vec![0.0]);
}


#[test]
fn test_sgd_negative_grad() {
    let w = Tensor::new(vec![1.0], vec![1, 1], true, vec![], None);
    w.tensor_data.borrow_mut().grad = vec![-1.0];

    let mut sgd = SGD::new(vec![w.tensor_data.clone()], 0.5, 0.0);
    sgd.backward_step();

    // v = -0.5, w = 1 - (-0.5) = 1.5
    assert_eq!(w.tensor_data.borrow().data, vec![1.5]);
    assert_eq!(sgd.velosity[0], vec![-0.5]);
}

#[test]
fn test_sgd_momentum_one() {
    let w = Tensor::new(vec![1.0], vec![1, 1], true, vec![], None);
    let mut sgd = SGD::new(vec![w.tensor_data.clone()], 0.5, 1.0);
    for _ in 0..3 {
        w.tensor_data.borrow_mut().grad = vec![1.0];
        sgd.backward_step();
    }
    assert_eq!(sgd.velosity[0], vec![1.5]);
    assert_eq!(w.tensor_data.borrow().data, vec![-2.0]);
}
