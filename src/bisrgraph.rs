use num::integer::binomial;

use crate::util::expected_success_distribution;
use std::collections::HashMap;
use std::hash::Hash;
use std::vec;

type Index = usize;
type Weight = f64;

#[derive(Debug)]
pub(crate) struct BiSRGraph {
    u: Index,
    v_lambda: Vec<Weight>,
    #[allow(dead_code)]
    u_adj: Vec<Vec<Index>>,
    v_adj: Vec<Vec<Index>>,
}

impl BiSRGraph {
    pub fn from_edge<Key>(edges: Vec<(Key, Key)>, v_weight: HashMap<Key, Weight>) -> Self
    where
        Key: Hash + std::cmp::Eq + Clone + std::fmt::Debug + std::fmt::Display,
    {
        let mut u_key = Vec::new();
        let mut v_key = Vec::new();
        let mut u_key2i = HashMap::new();
        let mut v_key2i = HashMap::new();

        let mut u_adj = Vec::new();
        let mut v_adj = Vec::new();

        let mut v_lambda = Vec::new();

        for edge in edges.iter() {
            let (u, v) = edge;
            let ui;
            let vi;
            match u_key2i.get(u) {
                Some(&i) => ui = i,
                None => {
                    ui = u_key.len();
                    u_key.push(u.clone());
                    u_key2i.insert(u.clone(), ui);
                    u_adj.push(Vec::new());
                }
            }

            match v_key2i.get(v) {
                Some(&i) => vi = i,
                None => {
                    vi = v_key.len();
                    v_key.push(v.clone());
                    v_key2i.insert(v.clone(), vi);
                    v_adj.push(Vec::new());

                    v_lambda.push(v_weight[v]);
                }
            }
            assert!(!u_adj[ui].contains(&vi), "diagnostic repeated edges");
            u_adj[ui].push(vi);

            assert!(!v_adj[vi].contains(&ui), "diagnostic repeated edges");
            v_adj[vi].push(ui);
        }

        for adj in u_adj.iter_mut() {
            adj.sort()
        }
        for adj in v_adj.iter_mut() {
            adj.sort()
        }

        BiSRGraph {
            u: u_key.len(),
            v_lambda,
            u_adj,
            v_adj,
        }
    }
}

impl BiSRGraph {
    #[allow(non_snake_case)]
    #[allow(dead_code)]
    pub fn OPT(self: &Self) -> f64 {
        todo!()
    }

    #[allow(non_snake_case)]
    pub fn ALG(self: &Self) -> f64 {
        use itertools::Itertools;
        if self.u == 0 {
            return 0.0;
        }
        let v0_adj = &self.v_adj[0];
        let lambda = self.v_lambda[0];
        let n = v0_adj.len();
        let distribution = expected_success_distribution(n, lambda);
        let mut exps = 0.;
        for k in 0..=n {
            exps += k as f64 * distribution[k];
            let pk = distribution[k] / binomial(n, k) as f64;
            for it in v0_adj.clone().into_iter().combinations(k) {
                let mut edges = vec![];
                let mut v_weight = HashMap::new();
                for vi in 1..self.v_adj.len() {
                    v_weight.insert(vi, self.v_lambda[vi]);
                    for &ui in &self.v_adj[vi] {
                        if !it.contains(&ui) {
                            edges.push((ui, vi));
                        }
                    }
                }
                let next_graph = Self::from_edge(edges, v_weight);
                exps += pk * next_graph.ALG();
            }
        }
        exps
    }
}
