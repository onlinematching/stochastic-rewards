#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unreachable_code)]

use anyhow::Result;
use once_cell::sync::Lazy;
use play::play::{filter_batch, iterate_batches};
use policy::policy::{policy_net, transmute_observation};
use std::collections::HashMap;
use std::sync::Mutex;
use tch::nn;
use tch::nn::{Module, OptimizerConfig};
use tch::Device;
use tch::Tensor;
use util::transmute_action_onehot;
mod bisrgraph;
mod env;
mod play;
mod policy;
mod test_edge;
mod test_net;
mod util;

static DEVICE: Lazy<Mutex<Device>> = Lazy::new(|| Device::cuda_if_available().into());

const PERCENTILE: i32 = 70;

pub fn run() -> Result<()> {
    let mut env = env::env::BiSRGraphGame::new();
    let vs = nn::VarStore::new(*DEVICE.lock().unwrap());
    let vs_ref_binding = vs.root();
    let policy_net = policy_net(&vs_ref_binding);
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    for epoch in 1..2 {
        let batch = iterate_batches(&mut env, &policy_net, 1);
        println!("{:?}", batch);
        let (obs_vec, act_vec, reward_bound, reward_mean) = filter_batch(batch, PERCENTILE);
        opt.zero_grad();
        let observation = obs_vec
            .iter()
            .map(|obs| transmute_observation(obs))
            .collect::<Vec<Tensor>>();
        let observation = Tensor::stack(&observation, 0);
        observation.print();
        println!("{:?}", observation.size());
        let action = act_vec
            .iter()
            .map(|act| transmute_action_onehot(act))
            .collect::<Vec<i64>>();
        let action = Tensor::of_slice(&action);
        action.print();
        let action_scores = policy_net.forward(&observation);
        // .cross_entropy_for_logits(&action);
        action_scores.print();
        let loss = action_scores.cross_entropy_for_logits(&action);
        opt.backward_step(&loss);
    }
    Ok(())
}

fn main() {
    println!("{:?}", *DEVICE.lock().unwrap());
    match run() {
        Ok(_) => {}
        Err(e) => {
            panic!("{:?}", e)
        }
    };
}
