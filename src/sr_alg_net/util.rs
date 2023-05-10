use std::collections::HashMap;

use super::env::{ActionProbSpace, ObservationSpace, Space};
use ndarray::Array;
use ndarray_rand::rand_distr::{Distribution, Uniform};
use onlinematching::{papers::stochastic_reward::graph::Prob, weightedbigraph::WBigraph};
use rand::{thread_rng, Rng};
use tch::Tensor;

pub const M: usize = 4;

pub const fn pow2(n: usize) -> usize {
    1 << n
}

pub fn bernoulli_trial(p: f64) -> bool {
    let mut rng = rand::thread_rng();
    rng.gen::<f64>() < p
}

pub fn generate_worst_edges() -> Vec<(usize, usize)> {
    match M {
        1 => {
            vec![(0, 0)]
        }
        2 => {
            vec![(0, 0), (1, 0), (1, 1)]
        }
        3 => {
            vec![(0, 0), (1, 0), (2, 0), (1, 1), (2, 2)]
        }
        4 => {
            //
            // u_adj: [[0], [0, 1], [0, 2], [0, 3]], v_adj: [[0, 1, 2, 3], [1], [2], [3]]
            vec![(0, 0), (1, 0), (1, 1), (2, 0), (2, 2), (3, 0), (3, 3)]
        }
        5 => {
            // u_adj: [[0, 1, 3], [0, 1, 4], [0, 1], [0], [0, 1, 2]],
            // v_adj: [[0, 1, 2, 3, 4], [0, 1, 2, 4], [4], [0], [1]]
            vec![
                (0, 0),
                (0, 1),
                (1, 3),
                (1, 0),
                (1, 1),
                (1, 4),
                (2, 0),
                (2, 1),
                (3, 0),
                (4, 0),
                (4, 1),
                (4, 2),
            ]
        }
        6 => {
            // u_adj: [[0, 1, 2], [0], [0, 1], [0, 1, 5], [0, 1, 4], [0, 1, 3]],
            // v_adj: [[0, 1, 2, 3, 4, 5], [0, 2, 3, 4, 5], [0], [5], [4], [3]]
            vec![
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (2, 0),
                (2, 1),
                (3, 0),
                (3, 1),
                (3, 5),
                (4, 0),
                (4, 1),
                (4, 4),
                (5, 0),
                (5, 1),
                (5, 3),
            ]
        }
        _ => panic!(),
    }
}

pub fn from_nonweight_edges(edges: &Vec<(usize, usize)>, m: usize) -> WBigraph<usize, f64> {
    let p = 1. / m as f64;
    let mut w_edges = Vec::new();
    for edge in edges {
        let (u, v) = *edge;
        for vi in (v * m)..((v + 1) * m) {
            w_edges.push(((u, vi), p))
        }
    }
    WBigraph::from_edges(&w_edges)
}

pub fn obser2tensor(obs: ObservationSpace) -> Tensor {
    let mut obs_trans: Vec<f32> = Vec::new();
    obs_trans.extend(obs.0.iter().map(|&a| a as f32));
    obs_trans.extend(obs.1.iter().map(|&a| a as f32));
    obs_trans.extend(obs.2.map(|a| match a {
        true => 1.,
        false => 0.,
    }));
    Tensor::of_slice(&obs_trans)
}

pub fn tensor2actprob(action: &Tensor) -> ActionProbSpace {
    let action_raw = Vec::<f32>::from(action.view(-1));
    let mut action = [0.; M];
    for i in 0..M {
        action[i] = action_raw[i] as f64;
    }
    (action,)
}

pub fn pre_deep_q_net_pretransmute(x: Space) -> Vec<f32> {
    let mut t: Vec<f32> = Vec::new();
    t.extend(x.0.unwrap().0.iter().map(|&a| a as f32));
    t.extend(x.0.unwrap().1.iter().map(|&a| a as f32));
    t.extend(x.0.unwrap().2.map(|a| match a {
        true => 1.,
        false => 0.,
    }));
    t.push(x.0.unwrap().3 as f32);
    match x.1 {
        Some(act) => t.push(act as f32),
        None => t.push(M as f32),
    }
    t
}

pub fn deep_q_net_pretransmute(x: Space) -> Tensor {
    Tensor::of_slice(&pre_deep_q_net_pretransmute(x))
}

pub fn sample<T>(my_vec: &Vec<T>) -> &T {
    let mut rng = thread_rng();
    let index = rng.gen_range(0..my_vec.len());
    &my_vec[index]
}

pub fn sample_from_softmax<const N: usize>(r: &[Prob; N]) -> usize {
    let arr = Array::from_iter(r.into_iter());
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
