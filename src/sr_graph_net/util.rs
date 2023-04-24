use crate::sr_graph_net::{
    bisrgraph::BiSRGraph,
    env::env::{ActionSpace, ObservationSpace, A},
    policy::policy::pow2,
};
use libm::{exp, pow};
use ndarray::{Array, Dim};
use ndarray_rand::rand_distr::Uniform;
use rand::prelude::*;
use std::collections::HashMap;

pub const M: usize = 3;

type Index = usize;
type Weight = f64;

#[allow(dead_code)]
pub fn sigmoid(x: f32) -> f32 {
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
        let tail_adj: &[A; M] = &graph_obs[step - 1];
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
    assert_eq!(graph_obs.1, M, "Agent has not yet been terminated");
    let graph_matrix = &graph_obs.0;
    let mut weights = vec![0; M];
    for i in 0..M {
        weights[i] = i;
    }
    let weights = array2weight(weights);
    let mut edges = Vec::new();
    for v in 0..M {
        for u in 0..M {
            if graph_matrix[v][u] == A::Success {
                edges.push((u, v));
            }
        }
    }

    BiSRGraph::from_edge(edges, weights)
}

fn sample_from_softmax(arr: &Array<&f32, Dim<[usize; 1]>>) -> usize {
    let mut rng = rand::thread_rng();
    let mut probabilities = arr.mapv(|x| x.exp());
    let sum = probabilities.sum();
    probabilities /= sum;
    let uniform = Uniform::new(0.0, 1.0);
    let mut acc = 0.0;
    for (index, value) in probabilities.iter().enumerate() {
        acc += *value;
        if uniform.sample(&mut rng) < acc {
            return index;
        }
    }
    arr.shape()[0] - 1
}

const LABELS: usize = pow2(M);

pub fn index2binary(index: usize) -> ActionSpace {
    let mut binary = [A::Fail; M];
    for i in 0..M {
        binary[M - 1 - i] = match (index & (1 << i)) != 0 {
            true => A::Success,
            false => A::Fail,
        };
    }
    binary
}

pub fn sampling_array(r: &[f32; LABELS]) -> ActionSpace {
    let arr = Array::from_iter(r.into_iter());
    let index = sample_from_softmax(&arr);
    index2binary(index)
}

pub fn percentile(mut data: Vec<f64>, percentile: f64) -> f64 {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ((percentile / 100.0) * (data.len() - 1) as f64).floor() as usize;
    let d = percentile / 100.0 * (data.len() - 1) as f64 - k as f64;
    if k >= data.len() - 1 {
        data[k]
    } else {
        data[k] * (1.0 - d) + data[k + 1] * d
    }
}

pub fn transmute_action_onehot(act_vec: &ActionSpace) -> i64 {
    let mut v = 0;
    for &act in act_vec {
        v *= 2;
        match act {
            A::Success => v += 1,
            A::Fail => {}
        }
    }
    v
}

#[cfg(test)]
mod tests_util {
    use super::{sampling_array, M};
    use crate::sr_graph_net::{env::env::A, util::percentile};

    #[test]
    fn test() {
        println!("Hello, world!");
    }

    #[test]
    /// when M = 4, then:
    /// obs_0 = [
    ///     [A::Success, A::Success, A::Success, A::Success],
    ///     [A::Fail, A::Success, A::Success, A::Success],
    ///     [A::Fail, A::Fail, A::Success, A::Success],
    ///     [A::Fail, A::Fail, A::Fail, A::Success],
    /// ];
    fn test_agent_generate_graph_m() {
        let mut obs_0 = [[A::Fail; M]; M];
        for i in 0..M {
            for j in i..M {
                obs_0[i][j] = A::Success
            }
        }
        let obs_1 = M;
        let g = super::agent_generate_graph(&(obs_0, obs_1));
        println!("{:?}", g);
        // BiSRGraph { u: 4, v_lambda: [1.0, 1.0, 1.0, 1.0], u_adj: [[0], [0, 1],
        //   [0, 1, 2], [0, 1, 2, 3]], v_adj: [[0, 1, 2, 3], [1, 2, 3], [2, 3], [3]] }
    }

    // #[test]
    // fn test_bernoulli_m5() {
    //     // let r = [0.1, 0.2, 0.8, 1.0, 1.1]; // value: 1.1 be InvalidProbability
    //     let r = [0.1, 0.2, 0.8, 1.0, 0.9];
    //     let sample = sampling_array(&r);
    //     println!("{:?}", sample);
    // }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p = percentile(data, 50.0);
        println!("{}", p); // 3.0
    }
}
