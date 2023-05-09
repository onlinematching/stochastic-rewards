use crate::sr_alg_net::awesome_alg::{deep_q_net, State, DEVICE};
use crate::sr_alg_net::env::{AdapticeAlgGame, ObservationSpace, Reward, Space};
use crate::sr_alg_net::play::{calculate_loss, play, Experience, ExperienceBuffer};
use anyhow::{Ok, Result};
use std::sync::Arc;
use tch::nn;
use tch::nn::OptimizerConfig;
use tch::Tensor;

const T: usize = 300;

pub fn run(batch_size: usize) -> Result<()> {
    let mut game: AdapticeAlgGame = AdapticeAlgGame::new();
    let vs: nn::VarStore = nn::VarStore::new(*DEVICE.lock().unwrap());
    let vs_ref_binding: nn::Path = vs.root();
    let deep_q_net = Arc::new(deep_q_net(&vs_ref_binding));
    let mut opt: nn::Optimizer = nn::Adam::default().build(&vs, 1e-3)?;
    let mut epoch = 0;
    let mut buffers = ExperienceBuffer::new();
    loop {
        epoch += 1;
        if epoch == T * batch_size {
            break;
        }
        if let Some(buffer) = play(&mut game, deep_q_net.clone(), State::Train) {
            buffers.buffer.extend(&buffer.buffer)
        }
        if epoch % batch_size == 0 {
            println!("\n epoch = {epoch} -------------------------- ");
            let spaces: Vec<Space> = buffers.bean(Experience::get_space);
            let rewards: Vec<Reward> = buffers.bean(|exp| exp.reward);
            let next_obs: Vec<Option<ObservationSpace>> = buffers.bean(|exp| exp.new_state);
            // println!("buffer = {:?}", buffer);
            opt.zero_grad();
            let loss: Tensor = calculate_loss(spaces, rewards, next_obs, deep_q_net.clone());
            // println!("loss = ");
            // loss.print();
            opt.backward_step(&loss);
            buffers.buffer.clear();
        }
    }
    println!("test: ---------------------------");
    println!(
        "buffer = {:?}",
        play(&mut game, deep_q_net.clone(), State::Infer)
    );
    Ok(())
}
