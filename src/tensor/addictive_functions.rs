use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::tensor::tensor::{Tensor, Operation};

pub fn im2col(tensor: Tensor, kernel_shape: Vec<usize>, stride: (usize, usize)) -> Tensor {
    let tensor_shape = tensor.tensor_data.borrow().shape.clone();
    let x_iter = (tensor_shape[tensor_shape.len() - 2] - kernel_shape[kernel_shape.len() - 2])
        / stride.0
        + 1;
    let y_iter = (tensor_shape[tensor_shape.len() - 1] - kernel_shape[kernel_shape.len() - 1])
        / stride.1
        + 1;

    let total_windows = tensor_shape[0] * x_iter * y_iter;
    let window_size = kernel_shape[1] * kernel_shape[2] * kernel_shape[3];
    let total_elements = total_windows * window_size;
    let mut output_data = vec![0.0f32; total_elements];

    let tensor_data = tensor.tensor_data.borrow().data.clone();
    output_data
        .par_chunks_mut(window_size)
        .enumerate()
        .for_each(|(window_idx, window)| {
            let image = window_idx / (x_iter * y_iter);
            let rem = window_idx % (x_iter * y_iter);
            let x = rem / y_iter;
            let y = rem % y_iter;

            let mut idx = 0;
            for layer in 0..tensor_shape[1] {
                for local_x in 0..kernel_shape[2] {
                    for local_y in 0..kernel_shape[3] {
                        let global_index =
                            image * tensor_shape[1] * tensor_shape[2] * tensor_shape[3]
                                + layer * tensor_shape[2] * tensor_shape[3]
                                + x * stride.0 * tensor_shape[3]
                                + local_x * tensor_shape[3]
                                + y * stride.1
                                + local_y;
                        window[idx] = tensor_data[global_index];
                        idx += 1;
                    }
                }
            }
        });

    let output_shape = vec![
        tensor_shape[0] * x_iter * y_iter,
        tensor_shape[1] * kernel_shape[2] * kernel_shape[3],
    ];
    let output = Tensor::new(output_data, output_shape, false, vec![], None);
    output
}

pub fn col2im(
    grad_col: Tensor,
    input_shape: Vec<usize>,
    kernel_shape: Vec<usize>,
    stride: (usize, usize),
) -> Tensor {
    // ===== Распаковка размеров =====
    let n = input_shape[0];
    let c = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];

    let kh = kernel_shape[2];
    let kw = kernel_shape[3];

    // ===== Сколько окон помещается =====
    // Формулы должны совпадать с прямым im2col — иначе индексация разъедется.
    let x_iter = (h - kh) / stride.0 + 1;
    let y_iter = (w - kw) / stride.1 + 1;

    // ===== Размер одного окна =====
    let window_size = c * kh * kw;

    // ===== Буфер под результат, изначально нули =====
    // Пиксели, не попавшие ни в одно окно, останутся нулевыми —
    // это корректно, у них градиент равен нулю.
    // Пиксели в перекрытиях будут накапливаться суммированием.
    let total = n * c * h * w;
    let mut grad_input = vec![0.0f32; total];

    // ===== Забираем данные градиента =====
    let grad_data = grad_col.tensor_data.borrow().data.clone();

    // ===== Параллельно по окнам (как в прямом im2col) =====
    //
    // Проблема с гонками: разные окна пишут в одни и те же ячейки `grad_input`
    // при stride < kernel_size. Поэтому параллелить по окнам напрямую нельзя
    // без атомиков. Простой безопасный вариант — обычный цикл.
    //
    // Если нужен параллелизм — можно распараллелить по батчам `image`
    // (у каждого изображения свой диапазон ячеек в `grad_input`, гонок нет),
    // либо использовать `Vec<AtomicF32>` / reduction по потокам.
    for window_idx in 0..(n * x_iter * y_iter) {
        // Разворачиваем линейный индекс окна в (image, x, y)
        let image = window_idx / (x_iter * y_iter);
        let rem = window_idx % (x_iter * y_iter);
        let x = rem / y_iter;
        let y = rem % y_iter;

        // Начало строки в grad_col, соответствующей этому окну
        let col_offset = window_idx * window_size;

        // Обходим окно в том же порядке, что и в im2col:
        // layer (C) → local_x (kH) → local_y (kW)
        let mut idx = 0;
        for layer in 0..c {
            for local_x in 0..kh {
                for local_y in 0..kw {
                    // Глобальный индекс пикселя в grad_input
                    let global_index =
                        image * c * h * w
                            + layer * h * w
                            + (x * stride.0 + local_x) * w
                            + (y * stride.1 + local_y);

                    // Накопление: +=, а не =, потому что при перекрытиях
                    // один пиксель получает вклад от нескольких окон.
                    grad_input[global_index] += grad_data[col_offset + idx];
                    idx += 1;
                }
            }
        }
    }

    let output = Tensor::new(grad_input, input_shape, false, vec![], None);
    output
}

/// Печатает статистику градиента: среднее, минимум, максимум, дисперсию.
/// Один проход по срезу, f64 для точности сумм.
pub fn print_grad_stats(name: &str, g: &[f32]) {
    if g.is_empty() {
        println!("{:>12}: (пусто)", name);
        return;
    }

    let n = g.len() as f64;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for &v in g {
        // NaN/Inf пропускаем, чтобы не портить min/max/mean
        if !v.is_finite() {
            continue;
        }
        let vf = v as f64;
        sum += vf;
        sum_sq += vf * vf;
        if v < min { min = v; }
        if v > max { max = v; }
    }

    let mean = sum / n;
    let var = (sum_sq / n) - mean * mean;
    let var = if var < 0.0 { 0.0 } else { var }; // защита от -1e-15

    println!(
        "{:>12?}: mean={:>+11.4e}  min={:>+11.4e}  max={:>+11.4e}  var={:>11.4e}",
        name, mean, min, max, var
    );
}


pub fn op_name(op: &Option<Operation>) -> &'static str {
    match op {
        None => "None",
        Some(Operation::Matmul) => "Matmul",
        Some(Operation::Add) => "Add",
        Some(Operation::ReLu) => "ReLu",
        Some(Operation::MSE) => "MSE",
        Some(Operation::Sigmoid) => "Sigmoid",
        Some(Operation::Tanh) => "Tanh",
        Some(Operation::CrossEntropyWithSoftmax) => "CrossEntropyWithSoftmax",
        Some(Operation::Conv2d(..)) => "Conv2d",
        Some(Operation::Padding(..)) => "Padding",
        Some(Operation::Reshape) => "Reshape",
        Some(Operation::MaxPool(..)) => "MaxPool",
    }
}