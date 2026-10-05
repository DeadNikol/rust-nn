//! Крейт для графа вычислений. Содержит только определение структуры и создание графа.
//! 
use std::{cell::RefCell, collections::HashSet, rc::Rc};
use crate::tensor::tensor::TensorData;

/// Структура графа. В себе при создании должна содержать только один тензор, от которого и будет искать граф вычислений
pub struct Graph {
    nodes: Vec<Rc<RefCell<TensorData>>>, // Хранит изменяемые ссылки на родителей, которые он и будет менять при слове backward
}

impl Graph {
    /// Возвращает новый граф с единственным значением внутри
    pub fn new(node: Rc<RefCell<TensorData>>) -> Self {
        Graph { nodes: vec![node] }
    }
    /// Создаёт граф вычислений через поиск в глубину (dfs)
    pub fn build_topo(&self) -> Vec<Rc<RefCell<TensorData>>> {
        let mut visited = HashSet::new();
        let mut topo = Vec::new();

        for node in &self.nodes {
            Self::dfs(node, &mut visited, &mut topo);
        }
        topo
    }
    // Алгоритм поиска в глубину
    fn dfs(
        node: &Rc<RefCell<TensorData>>,
        visited: &mut HashSet<*const TensorData>,
        topo: &mut Vec<Rc<RefCell<TensorData>>>,
    ) {
        let ptr = Rc::as_ptr(node) as *const TensorData; // получаем сырой указатель на TensorData
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        // Рекурсивно обходим родителей
        for parent in &node.borrow().parents {
            Self::dfs(parent, visited, topo);
        }

        topo.push(node.clone());
    }
}
