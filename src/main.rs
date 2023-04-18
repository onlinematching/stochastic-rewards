#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unreachable_code)]

use anyhow::Result;
use once_cell::sync::Lazy;
use sr_graph_net::play::play::{filter_batch, iterate_batches};
use sr_graph_net::policy::policy::{policy_net, transmute_observation};
use std::collections::HashMap;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use tch::nn;
use tch::nn::{Module, OptimizerConfig};
use tch::Device;
use tch::Tensor;
use sr_graph_net::util::transmute_action_onehot;
mod sr_graph_net;

static DEVICE: Lazy<Mutex<Device>> = Lazy::new(|| Device::cuda_if_available().into());

const PERCENTILE: f64 = 0.3;

pub fn run() -> Result<()> {
    let mut env = sr_graph_net::env::env::BiSRGraphGame::new();
    let vs = nn::VarStore::new(*DEVICE.lock().unwrap());
    let vs_ref_binding = vs.root();
    let policy_net = policy_net(&vs_ref_binding);
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    for epoch in 1..10000 {
        let batch = iterate_batches(&mut env, &policy_net, 1024);
        // println!("{:?}", batch);
        let (obs_vec, act_vec, reward_bound, reward_mean, reward_lowest) =
            filter_batch(batch, PERCENTILE);
        opt.zero_grad();
        let observation = obs_vec
            .iter()
            .map(|obs| transmute_observation(obs))
            .collect::<Vec<Tensor>>();
        let observation = Tensor::stack(&observation, 0);
        // observation.print();
        println!("{:?}", observation.size());
        let action = act_vec
            .iter()
            .map(|act| transmute_action_onehot(act))
            .collect::<Vec<i64>>();
        let action = Tensor::of_slice(&action);
        // action.print();
        let action_scores = policy_net.forward(&observation);
        // action_scores.print();
        let loss = action_scores.cross_entropy_for_logits(&action);
        println!(
            "reward_lowest = {:?}, reward_bound = {:?}, reward_mean = {:?}",
            reward_lowest, reward_bound, reward_mean
        );
        loss.print();
        opt.backward_step(&loss);
        if epoch % 40 == 0 {
            thread::sleep(Duration::from_secs(4));
        }
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
