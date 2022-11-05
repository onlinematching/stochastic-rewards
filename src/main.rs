use std::collections::HashMap;
use std::hash::Hash;

type Index = usize;
type Weight = f64;

struct BiSRGraph {
    u: Index,
    v_prob: Vec<Weight>,
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

        let mut v_prob = Vec::new();

        for edge in edges {
            let (u, v) = edge;
            if !u_key.contains(&u) {
                let i = v_key.len();
                u_key.push(u.clone());
                u_key2i.insert(u.clone(), i);
                u_adj.push(Vec::new());
            }
            if !v_key.contains(&v) {
                let i = v_key.len();
                v_key.push(v.clone());
                v_key2i.insert(v.clone(), i);
                v_adj.push(Vec::new());

                // match v_weight.get(&u) {
                //     Some(weight) => v_prob.push(*weight),
                //     None => panic!("v_weight don't contain the key {}", u),
                // }

                assert_eq!(
                    v_weight[i].0, v,
                    "the edges' order don't follow the v weight sequence, v_weight = {}, v = {}",
                    v_weight[i].0, v
                );
                v_prob.push(v_weight[i].1);
            }
            let ui = *u_key2i.get(&u).unwrap();
            let vi = *v_key2i.get(&v).unwrap();
            
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
            v_prob,
            u_adj,
            v_adj,
        }
    }
}

fn main() {
    println!("Hello, world!");
}
