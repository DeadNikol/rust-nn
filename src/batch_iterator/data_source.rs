use crate::tensor::tensor::{Tensor, TensorData};
use std::cell::RefCell;
use std::fs;
use std::path::Path;
use std::rc::Rc;

#[derive(Clone)]
pub enum DataSource {
    Tensor(Rc<RefCell<TensorData>>),         // готовые данные
    Images(Vec<String>, Option<(f32, f32)>), // пути к картинкам
}

impl DataSource {
    pub fn paths_from_folders(paths: Vec<String>) -> Self {
        let mut all_paths: Vec<String> = Vec::new();

        for folder in &paths {
            let folder_path = Path::new(folder);
            if !folder_path.is_dir() {
                eprintln!("WARNING: {:?} is not dir. Skipping...", folder_path);
                continue;
            }

            let entries = match fs::read_dir(folder_path) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("WARNING:  can't read {:?}: {:?}", folder_path, e);
                    continue;
                }
            };

            let mut folder_paths: Vec<String> = Vec::new();
            for entry in entries {
                let entry = match entry {
                    Ok(e) => e,
                    Err(e) => {
                        eprintln!("⚠️ ошибка чтения элемента в {}: {}", folder, e);
                        continue;
                    }
                };

                let path = entry.path();
                println!("{:?}", path);
                if !path.is_file() {
                    continue;
                }

                let ext = match path.extension() {
                    Some(e) => e.to_string_lossy().to_lowercase(),
                    None => continue,
                };
                if ext == "png" || ext == "jpg" || ext == "jpeg" {
                    folder_paths.push(path.to_string_lossy().to_string());
                }
            }

            folder_paths.sort();
            all_paths.extend(folder_paths);
        }

        DataSource::Images(all_paths, None)
    }

    pub fn from_folders_with_classes(folders: Vec<(String, usize)>) -> (Self, Vec<usize>) {
        let mut all_paths: Vec<String> = Vec::new();
        let mut all_labels: Vec<usize> = Vec::new();

        for (folder, class_idx) in &folders {
            let folder_path = Path::new(folder);
            if !folder_path.is_dir() {
                eprintln!("⚠️ {} не является папкой, пропускаю", folder);
                continue;
            }

            let entries = match fs::read_dir(folder_path) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("⚠️ не удалось прочитать {}: {}", folder, e);
                    continue;
                }
            };

            let mut folder_paths: Vec<String> = Vec::new();

            for entry in entries {
                let entry = match entry {
                    Ok(e) => e,
                    Err(e) => {
                        eprintln!("⚠️ ошибка чтения элемента в {}: {}", folder, e);
                        continue;
                    }
                };

                let path = entry.path();
                if !path.is_file() {
                    continue;
                }

                let ext = match path.extension() {
                    Some(e) => e.to_string_lossy().to_lowercase(),
                    None => continue,
                };

                if ext == "png" || ext == "jpg" || ext == "jpeg" {
                    folder_paths.push(path.to_string_lossy().to_string());
                }
            }

            folder_paths.sort();
            let count = folder_paths.len();

            // Пути — в общий список
            all_paths.extend(folder_paths);
            // Метки — тот же класс, повторённый `count` раз
            all_labels.extend(std::iter::repeat(*class_idx).take(count));
        }

        (DataSource::Images(all_paths, None), all_labels)
    }

    pub fn one_hot(labels: &[usize], num_classes: usize) -> Self {
        let n = labels.len();
        let mut data = vec![0.0; n * num_classes];

        for (i, &l) in labels.iter().enumerate() {
            assert!(
                l < num_classes,
                "label {} >= num_classes {}",
                l,
                num_classes
            );
            data[i * num_classes + l] = 1.0;
        }

        let tensor = Tensor::new(data, vec![n, num_classes], false, vec![], None);
        Self::Tensor(tensor.tensor_data.clone())
    }

    pub fn len(&self) -> usize {
        match self {
            DataSource::Tensor(t) => t.borrow().shape[0],
            DataSource::Images(paths, _) => paths.len(),
        }
    }

    pub fn sample_size(&self) -> usize {
        match self {
            DataSource::Tensor(t) => t.borrow().shape[1..].iter().product(),
            DataSource::Images(paths, _) => {
                if paths.is_empty() {
                    return 0;
                }

                let img = Tensor::from_path(&paths[0]).unwrap();
                img.tensor_data.borrow().shape[1..].iter().product()
            }
        }
    }

    pub fn sample_shape(&self) -> Vec<usize> {
        match self {
            DataSource::Tensor(t) => t.borrow().shape[1..].to_vec(),
            DataSource::Images(paths, _) => {
                if paths.is_empty() {
                    return vec![];
                }
                let img = Tensor::from_path(&paths[0]).unwrap();
                img.tensor_data.borrow().shape.clone()
            }
        }
    }

    pub fn get_sample(&self, idx: usize) -> Vec<f32> {
        match self {
            DataSource::Tensor(t) => {
                let data = t.borrow();
                let sample_size: usize = data.shape[1..].iter().product();
                let start = idx * sample_size;
                data.data[start..start + sample_size].to_vec()
            }
            DataSource::Images(paths, normalize) => {
                let img = Tensor::from_path(&paths[idx]).unwrap();
                let mut data = img.tensor_data.borrow().data.clone();

                if let Some((mean, std)) = normalize {
                    for v in data.iter_mut() {
                        *v = (*v - *mean) / *std;
                    }
                }

                data
            }
        }
    }

    pub fn normalized(self, mean: f32, std: f32) -> Self {
        match self {
            DataSource::Tensor(t) => DataSource::Tensor(t),   // без изменений
            DataSource::Images(paths, _ ) => DataSource::Images (
                paths,
                Some((mean, std)),
            ),
        }
    }
}
