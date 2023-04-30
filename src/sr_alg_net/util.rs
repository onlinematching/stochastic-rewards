use super::env::{ActionProbSpace, ObservationSpace, Space};
use ndarray::Array;
use ndarray_rand::rand_distr::{Distribution, Uniform};
use onlinematching::papers::stochastic_reward::graph::Prob;
use rand::{thread_rng, Rng};
use tch::Tensor;

pub const M: usize = 3;

pub const fn pow2(n: usize) -> usize {
    1 << n
}

pub fn bernoulli_trial(p: f64) -> bool {
    let mut rng = rand::thread_rng();
    rng.gen::<f64>() < p
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
