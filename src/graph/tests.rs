use super::graph::Graph;
use crate::tensor::tensor::Tensor;
use std::rc::Rc;

#[test]
fn test_graph_linear_size() {
    let a = Tensor::new(vec![1.0; 6], vec![2, 3], true, vec![], None);
    let b = Tensor::new(vec![1.0; 12], vec![3, 4], true, vec![], None);
    let out = a.matmul(&b);

    let graph = Graph::new(out.tensor_data.clone()).build_topo();
    assert_eq!(graph.len(), 3, "граф должен содержать 3 узла");

    // Проверим, что все три узла — те же Rc
    let has_a = graph.iter().any(|n| Rc::ptr_eq(n, &a.tensor_data));
    let has_b = graph.iter().any(|n| Rc::ptr_eq(n, &b.tensor_data));
    let has_out = graph.iter().any(|n| Rc::ptr_eq(n, &out.tensor_data));
    assert!(has_a, "a не в графе");
    assert!(has_b, "b не в графе");
    assert!(has_out, "out не в графе");
}

#[test]
fn test_graph_no_duplicates() {
    let a = Tensor::new(vec![1.0; 4], vec![2, 2], true, vec![], None);
    let out = a.matmul(&a); // a используется дважды

    let graph = Graph::new(out.tensor_data.clone()).build_topo();
    // Ожидаем: 2 узла — a и out (НЕ 3, потому что a один и тот же Rc)
    assert_eq!(graph.len(), 2, "a должен быть один раз, а не два");

    // Проверим, что a встречается ровно один раз
    let count_a = graph
        .iter()
        .filter(|n| Rc::ptr_eq(n, &a.tensor_data))
        .count();
    assert_eq!(count_a, 1, "a встречается {} раз, ожидалось 1", count_a);
}

#[test]
fn test_graph_conv2d_no_cycle() {
    let x = Tensor::new(vec![0.0; 9], vec![1, 1, 3, 3], true, vec![], None);
    let k = Tensor::new(vec![0.0; 4], vec![1, 1, 2, 2], true, vec![], None);
    let out = x.conv2d(&k, (1, 1));
    // Если граф зациклится, тест зависнет и упадёт по timeout
    let graph = Graph::new(out.tensor_data.clone()).build_topo();
    // Ожидаем 4 узла: x, k, temp_tensor, temp_kernel, out — но temp_* не в графе из-за operation
    println!("conv2d graph size = {}", graph.len());
    // Минимум — x, k, out
    assert!(graph.len() >= 3, "граф слишком маленький: {}", graph.len());
    // Должно быть в графе 5 узлов: 1 конечный и 4 родителя
    assert!(graph.len() <= 10, "граф слишком большой: {}", graph.len());
    // Проверим наличие x и k
    let has_x = graph.iter().any(|n| Rc::ptr_eq(n, &x.tensor_data));
    let has_k = graph.iter().any(|n| Rc::ptr_eq(n, &k.tensor_data));
    assert!(has_x, "x не в графе");
    assert!(has_k, "k не в графе");
}

#[test]
fn test_graph_topological_order() {
    let a = Tensor::new(vec![1.0; 6], vec![2, 3], true, vec![], None);
    let b = Tensor::new(vec![1.0; 12], vec![3, 4], true, vec![], None);
    let out = a.matmul(&b);
    let graph = Graph::new(out.tensor_data.clone()).build_topo();
    let pos_a = graph
        .iter()
        .position(|n| Rc::ptr_eq(n, &a.tensor_data))
        .unwrap();
    let pos_b = graph
        .iter()
        .position(|n| Rc::ptr_eq(n, &b.tensor_data))
        .unwrap();
    let pos_out = graph
        .iter()
        .position(|n| Rc::ptr_eq(n, &out.tensor_data))
        .unwrap();
    assert!(
        pos_a < pos_out,
        "a (idx {}) должен быть до out (idx {})",
        pos_a,
        pos_out
    );
    assert!(
        pos_b < pos_out,
        "b (idx {}) должен быть до out (idx {})",
        pos_b,
        pos_out
    );
}
#[test]
fn test_graph_chain_size() {
    let a = Tensor::new(vec![1.0; 6], vec![2, 3], true, vec![], None);
    let b = Tensor::new(vec![1.0; 6], vec![3, 2], true, vec![], None);
    let c = Tensor::new(vec![1.0; 4], vec![2, 2], true, vec![], None);
    let out = a.matmul(&b).add(&c);
    let graph = Graph::new(out.tensor_data.clone()).build_topo();
    assert_eq!(
        graph.len(),
        5,
        "граф должен содержать 5 узлов, а не {}",
        graph.len()
    );
}

#[test]
fn test_graph_conv_relu_pool() {
    let x = Tensor::new(vec![0.0; 9], vec![1, 1, 3, 3], true, vec![], None);
    let k = Tensor::new(vec![0.0; 4], vec![1, 1, 2, 2], true, vec![], None);
    let out = x.conv2d(&k, (1, 1)).relu().max_pool((2, 2), (1, 1));
    let graph = Graph::new(out.tensor_data.clone()).build_topo();
    println!("conv+relu+pool graph size = {}", graph.len());
    // Минимум: x, k, conv_out, relu_out, pool_out
    assert!(graph.len() >= 5, "граф слишком маленький: {}", graph.len());
    // x, k должны быть внутри
    let has_x = graph.iter().any(|n| Rc::ptr_eq(n, &x.tensor_data));
    let has_k = graph.iter().any(|n| Rc::ptr_eq(n, &k.tensor_data));
    assert!(has_x);
    assert!(has_k);
}

#[test]
fn test_graph_mse_linear() {
    let w = Tensor::new(
        vec![
            1., 0., 0., // row 0
            0., 1., 0.,
        ], // row 1
        vec![2, 3],
        true,
        vec![],
        None,
    );
    let b = Tensor::new(vec![0., 0., 0.], vec![1, 3], true, vec![], None);
    let x = Tensor::new(vec![1., 2.], vec![1, 2], true, vec![], None);
    let target = Tensor::new(vec![0., 0., 0.], vec![1, 3], false, vec![], None);
    let pred = x.matmul(&w).add(&b); // [1, 2] @ [2, 3] = [1, 3]
    {
        let p = pred.tensor_data.borrow();
        assert_eq!(p.shape, vec![1, 3]);
        assert_eq!(p.data, vec![1., 2., 0.]);
    }
    let loss = pred.mse(&target);
    let loss_val = loss.tensor_data.borrow().data[0];
    assert!((loss_val - 5.0 / 3.0).abs() < 1e-6, "loss = {}", loss_val);

    loss.backward();
    let w_grad = w.tensor_data.borrow().grad.clone();
    let expected_w = vec![2. / 3., 4. / 3., 0., 4. / 3., 8. / 3., 0.];
    for (i, (g, e)) in w_grad.iter().zip(expected_w.iter()).enumerate() {
        assert!((g - e).abs() < 1e-6, "W.grad[{}]: {} != {}", i, g, e);
    }
    let b_grad = b.tensor_data.borrow().grad.clone();
    assert!((b_grad[0] - 2. / 3.).abs() < 1e-6);
    assert!((b_grad[1] - 4. / 3.).abs() < 1e-6);
    assert!((b_grad[2] - 0.).abs() < 1e-6);
    let x_grad = x.tensor_data.borrow().grad.clone();
    assert!((x_grad[0] - 2. / 3.).abs() < 1e-6);
    assert!((x_grad[1] - 4. / 3.).abs() < 1e-6);
}

#[test]
fn test_graph_mse_two_layers_relu() {
    let w1 = Tensor::new(vec![1., 0., 0., 1.], vec![2, 2], true, vec![], None);
    let b1 = Tensor::new(vec![0., 0.], vec![1, 2], true, vec![], None);
    let w2 = Tensor::new(vec![1., 0., 0., 1.], vec![2, 2], true, vec![], None);
    let b2 = Tensor::new(vec![0., 0.], vec![1, 2], true, vec![], None);
    let x = Tensor::new(vec![-1., 2.], vec![1, 2], true, vec![], None);
    let target = Tensor::new(vec![0., 0.], vec![1, 2], false, vec![], None);

    let z1 = x.matmul(&w1).add(&b1); // [-1, 2]
    let a1 = z1.relu(); // [0, 2]
    let z2 = a1.matmul(&w2).add(&b2); // [0, 2]
    let loss = z2.mse(&target); // (0+4)/2 = 2
    assert!((loss.tensor_data.borrow().data[0] - 2.0).abs() < 1e-6);
    loss.backward();
    assert_eq!(b2.tensor_data.borrow().grad, vec![0., 2.]);
    assert_eq!(w2.tensor_data.borrow().grad, vec![0., 0., 0., 4.]);
    assert_eq!(b1.tensor_data.borrow().grad, vec![0., 2.]);
    assert_eq!(w1.tensor_data.borrow().grad, vec![0., -2., 0., 4.]);
    assert_eq!(x.tensor_data.borrow().grad, vec![0., 2.]);
}

#[test]
fn test_graph_conv2d_mse() {
    let x = Tensor::new(vec![1., 2., 3., 4.], vec![1, 1, 2, 2], true, vec![], None);
    let k = Tensor::new(vec![1., 1., 1., 1.], vec![1, 1, 2, 2], true, vec![], None);
    let target = Tensor::new(vec![0.], vec![1, 1, 1, 1], false, vec![], None);
    let out = x.conv2d(&k, (1, 1));
    assert_eq!(out.tensor_data.borrow().data, vec![10.]);
    let loss = out.mse(&target);
    assert!((loss.tensor_data.borrow().data[0] - 100.).abs() < 1e-6);
    loss.backward();
    assert_eq!(x.tensor_data.borrow().grad, vec![20., 20., 20., 20.]);
    assert_eq!(k.tensor_data.borrow().grad, vec![20., 40., 60., 80.]);
}

#[test]
fn test_graph_cross_entropy_linear() {
    let w = Tensor::new(vec![0., 0., 0., 0.], vec![2, 2], true, vec![], None);
    let b = Tensor::new(vec![0., 0.], vec![1, 2], true, vec![], None);
    let x = Tensor::new(vec![1., 1.], vec![1, 2], true, vec![], None);
    let target = Tensor::new(vec![1., 0.], vec![1, 2], false, vec![], None);

    let logits = x.matmul(&w).add(&b); // [0, 0]
    let loss = logits.cross_entropy_with_softmax(&target);
    let loss_val = loss.tensor_data.borrow().data[0];
    assert!(
        (loss_val - (2.0_f32).ln()).abs() < 1e-6,
        "loss = {}",
        loss_val
    );

    loss.backward();
    let b_grad = b.tensor_data.borrow().grad.clone();
    assert!((b_grad[0] - (-0.5)).abs() < 1e-6);
    assert!((b_grad[1] - 0.5).abs() < 1e-6);
    let w_grad = w.tensor_data.borrow().grad.clone();
    let expected_w = vec![-0.5, 0.5, -0.5, 0.5];
    for (i, (g, e)) in w_grad.iter().zip(expected_w.iter()).enumerate() {
        assert!((g - e).abs() < 1e-6, "W.grad[{}]: {} != {}", i, g, e);
    }
    assert_eq!(x.tensor_data.borrow().grad, vec![0., 0.]);
}

#[test]
fn test_graph_cross_entropy_two_layers_relu() {
    let w1 = Tensor::new(vec![1., 0., 0., 1.], vec![2, 2], true, vec![], None);
    let b1 = Tensor::new(vec![0., 0.], vec![1, 2], true, vec![], None);
    let w2 = Tensor::new(vec![1., 0., 0., 1.], vec![2, 2], true, vec![], None);
    let b2 = Tensor::new(vec![0., 0.], vec![1, 2], true, vec![], None);
    let x = Tensor::new(vec![1., -1.], vec![1, 2], true, vec![], None);
    let target = Tensor::new(vec![1., 0.], vec![1, 2], false, vec![], None);

    let logits = x.matmul(&w1).add(&b1).relu().matmul(&w2).add(&b2);
    {
        let l = logits.tensor_data.borrow();
        assert_eq!(l.data, vec![1., 0.]);
    }
    let loss = logits.cross_entropy_with_softmax(&target);
    let loss_val = loss.tensor_data.borrow().data[0];
    assert!((loss_val - 0.313).abs() < 1e-2, "loss = {}", loss_val);

    loss.backward();
    let b2_grad = b2.tensor_data.borrow().grad.clone();
    assert!((b2_grad[0] - (-0.269)).abs() < 1e-2);
    assert!((b2_grad[1] - 0.269).abs() < 1e-2);
    let b1_grad = b1.tensor_data.borrow().grad.clone();
    assert!(b1_grad[1].abs() < 1e-2, "b1.grad[1] должен быть ~0");
    let x_grad = x.tensor_data.borrow().grad.clone();
    assert!(x_grad[1].abs() < 1e-2, "x.grad[1] должен быть ~0");
}
