use super::batch_iterator::{BatchIterator, DataSource};
use crate::tensor::tensor::Tensor;

// ============================================================
// ХЕЛПЕР
// ============================================================

/// Возвращает пути ко всем 10 картинкам в images/.
fn all_image_paths() -> Vec<String> {
    (0..10)
        .map(|i| format!("src/batch_iterator/images/0_{:06}.png", i))
        .collect()
}

/// Возвращает форму одной картинки — считает через Tensor::from_path.
fn image_shape() -> Vec<usize> {
    let t = Tensor::from_path("src/batch_iterator/images/0_000000.png")
        .expect("картинка 0_000000.png должна существовать");
    t.tensor_data.borrow().shape.clone()
}

// ============================================================
// ЧАСТЬ 1: DataSource::Images — метаданные
// ============================================================

#[test]
fn test_images_len() {
    let ds = DataSource::Images(all_image_paths());
    assert_eq!(ds.len(), 10);
}

#[test]
fn test_images_sample_shape_matches_single_tensor() {
    let ds = DataSource::Images(all_image_paths());
    let expected = image_shape();
    assert_eq!(ds.sample_shape(), expected);
}

#[test]
fn test_images_sample_size_matches_shape_product() {
    let ds = DataSource::Images(all_image_paths());
    let shape = ds.sample_shape();
    let expected: usize = shape.iter().product();
    assert_eq!(ds.sample_size(), expected);
}

#[test]
fn test_images_get_sample_matches_from_path() {
    let ds = DataSource::Images(all_image_paths());
    let sample = ds.get_sample(0);

    let direct = Tensor::from_path("src/batch_iterator/images/0_000000.png").unwrap();
    let expected = direct.tensor_data.borrow().data.clone();

    assert_eq!(
        sample, expected,
        "get_sample(0) должен совпасть с Tensor::from_path"
    );
}

#[test]
fn test_images_get_sample_each_index() {
    let ds = DataSource::Images(all_image_paths());

    for i in 0..10 {
        let sample = ds.get_sample(i);
        let direct =
            Tensor::from_path(&format!("src/batch_iterator/images/0_{:06}.png", i)).unwrap();
        let expected = direct.tensor_data.borrow().data.clone();
        assert_eq!(sample, expected, "get_sample({}) не совпал", i);
    }
}

#[test]
fn test_images_pixels_in_normalized_range() {
    // Tensor::from_path нормализует пиксели в [0, 1]
    let ds = DataSource::Images(all_image_paths());
    let sample = ds.get_sample(0);
    for (i, v) in sample.iter().enumerate() {
        assert!(*v >= 0.0 && *v <= 1.0, "пиксель[{}] = {} вне [0, 1]", i, v);
    }
}

// ============================================================
// ЧАСТЬ 2: BatchIterator с картинками
// ============================================================

#[test]
fn test_batch_iterator_images_num_batches() {
    let paths = all_image_paths();
    let bi = BatchIterator::new(
        DataSource::Images(paths.clone()),
        DataSource::Images(paths),
        3,
    );
    // 10 / 3 = 4 батча (последний неполный)
    assert_eq!(bi.num_batches(), 4);
}

#[test]
fn test_batch_iterator_images_shapes() {
    let paths = all_image_paths();
    let mut bi = BatchIterator::new(
        DataSource::Images(paths.clone()),
        DataSource::Images(paths),
        3,
    );

    let sample_shape = image_shape(); // [C, H, W]

    // Батч 1: 3 сэмпла
    let (bx1, by1) = bi.next().unwrap();
    let mut expected_shape1 = vec![3];
    expected_shape1.extend_from_slice(&sample_shape);
    assert_eq!(bx1.tensor_data.borrow().shape, expected_shape1);
    assert_eq!(by1.tensor_data.borrow().shape, expected_shape1);

    // Батч 2: 3 сэмпла
    let (bx2, _) = bi.next().unwrap();
    assert_eq!(bx2.tensor_data.borrow().shape[0], 3);

    // Батч 3: 3 сэмпла
    let (bx3, _) = bi.next().unwrap();
    assert_eq!(bx3.tensor_data.borrow().shape[0], 3);

    // Батч 4: 1 сэмпл (неполный)
    let (bx4, _) = bi.next().unwrap();
    let mut expected_shape4 = vec![1];
    expected_shape4.extend_from_slice(&sample_shape);
    assert_eq!(bx4.tensor_data.borrow().shape, expected_shape4);

    // Конец
    assert!(bi.next().is_none());
}

#[test]
fn test_batch_iterator_images_covers_all_samples() {
    // Ключевой тест: батчи вместе покрывают ровно все 10 сэмплов
    let paths = all_image_paths();
    let mut bi = BatchIterator::new(
        DataSource::Images(paths.clone()),
        DataSource::Images(paths),
        4,
    );

    let mut total_samples = 0;
    while let Some((bx, _)) = bi.next() {
        let batch_len = bx.tensor_data.borrow().shape[0];
        total_samples += batch_len;
    }
    assert_eq!(total_samples, 10, "суммарно должно быть 10 сэмплов");
}

#[test]
fn test_batch_iterator_images_batch_content_matches_indices() {
    // Батч должен содержать данные в порядке indices[start..end]
    let paths = all_image_paths();
    let mut bi = BatchIterator::new(
        DataSource::Images(paths.clone()),
        DataSource::Images(paths.clone()),
        3,
    );

    // Первый батч: индексы 0, 1, 2 (по порядку, без shuffle)
    let (bx1, _) = bi.next().unwrap();
    let batch_data = bx1.tensor_data.borrow().data.clone();

    let sample_size = bi.x_source.sample_size();
    for i in 0..3 {
        let expected_sample = Tensor::from_path(&paths[i]).unwrap();
        let expected = expected_sample.tensor_data.borrow().data.clone();
        let got = &batch_data[i * sample_size..(i + 1) * sample_size];
        assert_eq!(got, &expected[..], "сэмпл {} в батче не совпал", i);
    }
}

// ============================================================
// ЧАСТЬ 3: Shuffle
// ============================================================

#[test]
fn test_images_initial_indices_sequential() {
    let paths = all_image_paths();
    let bi = BatchIterator::new(
        DataSource::Images(paths.clone()),
        DataSource::Images(paths),
        3,
    );
    assert_eq!(bi.indices, (0..10).collect::<Vec<_>>());
}

#[test]
fn test_images_shuffle_changes_order() {
    let paths = all_image_paths();
    let mut bi = BatchIterator::new(
        DataSource::Images(paths.clone()),
        DataSource::Images(paths),
        3,
    );

    let before = bi.indices.clone();
    bi.reset_indices();
    let after = bi.indices.clone();

    assert_ne!(before, after, "shuffle должен изменить порядок");
}

#[test]
fn test_images_shuffle_keeps_all_indices() {
    let paths = all_image_paths();
    let mut bi = BatchIterator::new(
        DataSource::Images(paths.clone()),
        DataSource::Images(paths),
        3,
    );

    bi.reset_indices();

    let mut sorted = bi.indices.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        (0..10).collect::<Vec<_>>(),
        "после shuffle все индексы на месте"
    );
}

#[test]
fn test_images_shuffle_resets_current() {
    let paths = all_image_paths();
    let mut bi = BatchIterator::new(
        DataSource::Images(paths.clone()),
        DataSource::Images(paths),
        3,
    );

    // Прокручиваем всё
    while bi.next().is_some() {}
    assert!(bi.next().is_none());

    // Сброс
    bi.reset_indices();
    assert_eq!(bi.current, 0);
    assert!(
        bi.next().is_some(),
        "после reset должен снова отдавать батчи"
    );
}

#[test]
fn test_images_shuffle_batch_content_follows_new_indices() {
    // После shuffle батч должен идти в порядке новых indices
    let paths = all_image_paths();
    let mut bi = BatchIterator::new(
        DataSource::Images(paths.clone()),
        DataSource::Images(paths.clone()),
        3,
    );

    bi.reset_indices();
    let new_indices = bi.indices.clone();

    let (bx1, _) = bi.next().unwrap();
    let batch_data = bx1.tensor_data.borrow().data.clone();
    let sample_size = bi.x_source.sample_size();

    // Первый батч должен содержать paths[new_indices[0..3]]
    for i in 0..3 {
        let idx = new_indices[i];
        let expected_sample = Tensor::from_path(&paths[idx]).unwrap();
        let expected = expected_sample.tensor_data.borrow().data.clone();
        let got = &batch_data[i * sample_size..(i + 1) * sample_size];
        assert_eq!(
            got,
            &expected[..],
            "сэмпл {} (idx {}) не совпал после shuffle",
            i,
            idx
        );
    }
}

// ============================================================
// ЧАСТЬ 4: Граничные случаи
// ============================================================

#[test]
fn test_images_single_sample() {
    let paths = vec!["src/batch_iterator/images/0_000000.png".to_string()];
    let mut bi = BatchIterator::new(
        DataSource::Images(paths.clone()),
        DataSource::Images(paths),
        1,
    );

    assert_eq!(bi.num_batches(), 1);
    assert_eq!(bi.indices, vec![0]);

    let (bx, _) = bi.next().unwrap();
    assert_eq!(bx.tensor_data.borrow().shape[0], 1);
    assert!(bi.next().is_none());
}

#[test]
fn test_images_batch_bigger_than_data() {
    let paths = all_image_paths();
    let mut bi = BatchIterator::new(
        DataSource::Images(paths.clone()),
        DataSource::Images(paths),
        100, // batch_size > num_samples
    );

    assert_eq!(bi.num_batches(), 1);

    let (bx, _) = bi.next().unwrap();
    assert_eq!(
        bx.tensor_data.borrow().shape[0],
        10,
        "весь датасет в одном батче"
    );
    assert!(bi.next().is_none());
}

#[test]
fn test_images_empty_source() {
    let ds = DataSource::Images(vec![]);
    assert_eq!(ds.len(), 0);
    assert_eq!(ds.sample_size(), 0);
    assert_eq!(ds.sample_shape(), Vec::<usize>::new());
}

// ============================================================
// ЧАСТЬ 5: DataSource::Tensor — базовая проверка
// ============================================================

#[test]
fn test_tensor_source_len() {
    let x = Tensor::new(vec![0.0; 12], vec![4, 3], false, vec![], None);
    let ds = DataSource::Tensor(x.tensor_data.clone());
    assert_eq!(ds.len(), 4);
}

#[test]
fn test_tensor_source_sample_shape() {
    let x = Tensor::new(vec![0.0; 12], vec![4, 3], false, vec![], None);
    let ds = DataSource::Tensor(x.tensor_data.clone());
    assert_eq!(ds.sample_shape(), vec![3]);
}

#[test]
fn test_tensor_source_get_sample() {
    // [3, 2] = [[0,1],[2,3],[4,5]]
    let x = Tensor::new(
        vec![0., 1., 2., 3., 4., 5.],
        vec![3, 2],
        false,
        vec![],
        None,
    );
    let ds = DataSource::Tensor(x.tensor_data.clone());

    assert_eq!(ds.get_sample(0), vec![0., 1.]);
    assert_eq!(ds.get_sample(1), vec![2., 3.]);
    assert_eq!(ds.get_sample(2), vec![4., 5.]);
}

// ============================================================
// ЧАСТЬ 6: Смешанный источник (X=Images, Y=Tensor)
// ============================================================

#[test]
fn test_mixed_x_images_y_tensor() {
    // Реалистичный случай: картинки + метки
    let paths = all_image_paths();
    let labels = Tensor::new(
        vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
        vec![10, 1],
        false,
        vec![],
        None,
    );

    let mut bi = BatchIterator::new(
        DataSource::Images(paths),
        DataSource::Tensor(labels.tensor_data.clone()),
        3,
    );

    assert_eq!(bi.num_batches(), 4);

    let (bx, by) = bi.next().unwrap();
    assert_eq!(bx.tensor_data.borrow().shape[0], 3);
    assert_eq!(by.tensor_data.borrow().shape, vec![3, 1]);
    assert_eq!(by.tensor_data.borrow().data, vec![0., 1., 2.]);
}
