use std::sync::Arc;
use std::time::Duration;

use crate::sr_alg_net::awesome_alg::{deep_q_net, DEVICE};
use crate::sr_alg_net::env::AdapticeAlgGame;
use crate::sr_alg_net::play::{filter_batch, iterate_batches};
use crate::sr_alg_net::util::obser2tensor;
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
    let deep_q_net = Arc::new(deep_q_net(&vs_ref_binding));
    let mut opt: nn::Optimizer = nn::Adam::default().build(&vs, 1e-3)?;
    for epoch in 1..10000 {
        
    }
    Ok(())
}
