use crate::sr_alg_net::awesome_alg::{deep_q_net, DEVICE};
use crate::sr_alg_net::env::{AdapticeAlgGame, ObservationSpace, Reward, Space};
use crate::sr_alg_net::play::{calculate_loss, play, Experience};
use anyhow::Result;
use std::sync::Arc;
use tch::nn;
use tch::nn::OptimizerConfig;
use tch::Tensor;

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
            println!("\n epoch = {epoch}");
            let spaces: Vec<Space> = buffer.bean(Experience::get_space);
            let rewards: Vec<Reward> = buffer.bean(|exp| exp.reward);
            let done: Vec<bool> = buffer.bean(|exp| exp.done);
            let next_obs: Vec<Option<ObservationSpace>> = buffer.bean(|exp| exp.new_state);
            println!("buffer = {:?}", buffer);
            let loss = calculate_loss(spaces, rewards, next_obs, done, deep_q_net.clone());
            
        }
    }
}
