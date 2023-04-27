use super::env::{ActionProbSpace, ObservationSpace};
use ndarray::Array;
use ndarray_rand::rand_distr::{Distribution, Uniform};
use onlinematching::papers::stochastic_reward::graph::Prob;
use rand::Rng;
use tch::Tensor;

pub const M: usize = 3;

pub const fn pow2(n: usize) -> usize {
    1 << n
}

#[allow(dead_code)]
pub fn bernoulli_trial(p: f64) -> bool {
    let mut rng = rand::thread_rng();
    rng.gen::<f64>() < p
}

pub fn transmute_obs(obs: ObservationSpace) -> Tensor {
    let mut obs_transmute: Vec<f32> = Vec::new();
    obs_transmute.extend(obs.0.iter().map(|&a| a as f32));
    obs_transmute.extend(obs.1.iter().map(|&a| a as f32));
    obs_transmute.extend(obs.2.iter().map(|&a| a as f32));
    obs_transmute.extend(obs.3.map(|a| match a {
        true => 1.,
        false => 0.,
    }));
    Tensor::of_slice(&obs_transmute)
}

pub fn transmute_act(action: &Tensor) -> ActionProbSpace {
    let action_raw = Vec::<f32>::from(action.view(-1));
    let mut action = [0.; M];
    for i in 0..M {
        action[i] = action_raw[i] as f64;
    }
    (action,)
}

pub fn sample_from_softmax(r: &[Prob; M]) -> usize {
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
