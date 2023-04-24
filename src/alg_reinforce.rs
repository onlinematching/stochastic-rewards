use std::sync::Arc;
use std::time::Duration;

use crate::sr_alg_net::awesome_alg::{policy_net, DEVICE};
use crate::sr_alg_net::env::AdapticeAlgGame;
use crate::sr_alg_net::play::{filter_batch, iterate_batches};
use crate::sr_alg_net::util::transmute_obs;
use anyhow::{Ok, Result};
use std::thread;
use tch::nn;
use tch::nn::{Module, OptimizerConfig};
use tch::Tensor;

const PERCENTILE: f64 = 0.3;

pub fn run() -> Result<()> {
    let mut game: AdapticeAlgGame = AdapticeAlgGame::new();
    let vs: nn::VarStore = nn::VarStore::new(*DEVICE.lock().unwrap());
    let vs_ref_binding: nn::Path = vs.root();
    let policy_net = Arc::new(policy_net(&vs_ref_binding));
    let mut opt: nn::Optimizer = nn::Adam::default().build(&vs, 1e-3)?;
    for epoch in 1..10000 {
        let batch = iterate_batches(&mut game, policy_net.clone(), 1024);
        let (obs_vec, act_vec, reward_bound, reward_mean, reward_lowest) =
            filter_batch(batch, PERCENTILE);
        opt.zero_grad();
        let observation = obs_vec
            .iter()
            .map(|obs| transmute_obs(*obs))
            .collect::<Vec<Tensor>>();
        let observation: Tensor = Tensor::stack(&observation, 0);

        println!("{:?}", observation.size());
        let action = act_vec.iter().map(|&act| act as i64).collect::<Vec<i64>>();
        let action = Tensor::of_slice(&action);
        let action_scores: Tensor = policy_net.forward(&observation);
        // action_scores.print();
        let loss: Tensor = action_scores.cross_entropy_for_logits(&action);
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
