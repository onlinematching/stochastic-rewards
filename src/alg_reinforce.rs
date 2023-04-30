use crate::sr_alg_net::awesome_alg::{deep_q_net, DEVICE};
use crate::sr_alg_net::env::AdapticeAlgGame;
use crate::sr_alg_net::play::play;
use anyhow::Result;
use std::sync::Arc;
use tch::nn;
use tch::nn::OptimizerConfig;
use tch::Tensor;

const PERCENTILE: f64 = 0.3;

pub fn run() -> Result<()> {
    let mut game: AdapticeAlgGame = AdapticeAlgGame::new();
    let vs: nn::VarStore = nn::VarStore::new(*DEVICE.lock().unwrap());
    let vs_ref_binding: nn::Path = vs.root();
    let deep_q_net = Arc::new(deep_q_net(&vs_ref_binding));
    let mut opt: nn::Optimizer = nn::Adam::default().build(&vs, 1e-3)?;
    let mut epoch = 0;
    loop {
        epoch += 1;
        if let Some(buffer) = play(&mut game, deep_q_net.clone()) {
            let buffer = buffer.buffer;
            buffer;
        }
    }
}
