use crate::sr_alg_net::awesome_alg::{policy_net, DEVICE};
use crate::sr_alg_net::env::AdapticeAlgGame;
use anyhow::{Ok, Result};
use tch::nn;
use tch::nn::{Module, OptimizerConfig};
use tch::Device;
use tch::Tensor;

const PERCENTILE: f64 = 0.3;

pub fn run() -> Result<()> {
    let mut env: AdapticeAlgGame = AdapticeAlgGame::new();
    let vs: nn::VarStore = nn::VarStore::new(*DEVICE.lock().unwrap());
    let vs_ref_binding: nn::Path = vs.root();
    let policy_net = policy_net(&vs_ref_binding);
    let mut opt: nn::Optimizer = nn::Adam::default().build(&vs, 1e-3)?;
    for epoch in 1..10000 {
        
    }
    Ok(())
}
