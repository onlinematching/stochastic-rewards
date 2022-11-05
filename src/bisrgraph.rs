use std::collections::HashMap;
use std::hash::Hash;

type Index = usize;
type Weight = f64;

#[derive(Debug)]
pub(crate) struct BiSRGraph {
    u: Index,
    v_lambda: Vec<Weight>,
    u_adj: Vec<Vec<Index>>,
    v_adj: Vec<Vec<Index>>,
}

impl BiSRGraph {
    pub fn from_edge<Key>(edges: Vec<(Key, Key)>, v_weight: Vec<(Key, Weight)>) -> Self
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

        for edge in edges {
            let (u, v) = edge;
            let ui;
            let vi;
            match u_key2i.get(&u) {
                Some(&i) => ui = i,
                None => {
                    ui = u_key.len();
                    u_key.push(u.clone());
                    u_key2i.insert(u.clone(), ui);
                    u_adj.push(Vec::new());
                }
            }

            match v_key2i.get(&v) {
                Some(&i) => vi = i,
                None => {
                    vi = v_key.len();
                    v_key.push(v.clone());
                    v_key2i.insert(v.clone(), vi);
                    v_adj.push(Vec::new());

                    assert_eq!(
                        v_weight[vi].0, v,
                        "the edges' order don't follow the v weight sequence, v_weight = {}, v = {}",
                        v_weight[vi].0, v
                    );
                    v_lambda.push(v_weight[vi].1);
                }
            }

            u_adj[ui].push(vi);
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
