#![allow(dead_code)]

use crate::{
    bisrgraph::BiSRGraph,
    env::env::{ObservationSpace, A},
};
use libm::{exp, pow};
use std::collections::HashMap;

pub const M: usize = 4;

type Index = usize;
type Weight = f64;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn factorial(num: usize) -> usize {
    match num {
        0 => 1,
        1 => 1,
        _ => factorial(num - 1) * num,
    }
}

pub fn expected_success_distribution(n: Index, lambda: Weight) -> Vec<f64> {
    let mut distro = vec![];
    let mut s = 0.;
    let p = exp(-lambda);
    for k in 0..n {
        let pow_lambda = pow(lambda, k as f64);
        let fac_k = factorial(k) as f64;
        let t = p * pow_lambda / fac_k;
        s += t;
        distro.push(t);
    }
    distro.push(1. - s);
    distro
}

pub fn array2weight<T>(v: Vec<T>) -> HashMap<T, f64>
where
    T: std::hash::Hash + std::cmp::Eq,
{
    let mut map = HashMap::new();
    for k in v.into_iter() {
        map.insert(k, 1.);
    }
    map
}

pub fn check_symmetry_property(graph_obs: &[[A; M]; M], step: usize) -> bool {
    for i in 0..step {
        let fore_adj: &[A; M] = &graph_obs[i];
        let tail_adj: &[A; M] = &graph_obs[step];
        let mut contain = true;
        let mut disjoint = true;
        for j in 0..M {
            if fore_adj[j] == A::Success && tail_adj[j] == A::Success {
                disjoint = false;
            }
        }
        for j in 0..M {
            if tail_adj[j] == A::Success && fore_adj[j] == A::Fail {
                contain = false;
            }
        }
        if contain || disjoint {
            continue;
        }
        return false;
    }
    true
}

pub fn agent_generate_graph(graph_obs: &ObservationSpace) -> BiSRGraph {
    todo!()
}
