use std::sync::Arc;

use crate::sr_alg_net::env::Space;
use onlinematching::papers::stochastic_reward::graph::Prob;
use tch::nn::Module;

use super::env::{ActionSpace, AdapticeAlgGame, ObservationSpace};

const SEED: i64 = 42;
const DEBUG: bool = false;

#[derive(Clone, Debug)]
pub struct EpisodeStep {
    observation: ObservationSpace,
    action: ActionSpace,
}

#[derive(Debug)]
pub struct Episode {
    reward: f64,
    steps: Vec<EpisodeStep>,
}

pub fn iterate_batches(
    game: &mut AdapticeAlgGame,
    net: Arc<dyn Module>,
    batch_size: usize,
) -> Vec<Episode> {
    let mut batch: Vec<Episode> = vec![];
    let mut episode_reward: f64 = 0.;
    let mut episode_steps: Vec<EpisodeStep> = vec![];
    let mut obs: (ObservationSpace, f64, bool, bool) = game.reset(net.clone(), SEED);
    loop {
        let obs_old = obs.0;
        let online_adjacent = game.get_online_adjacent();
        let info: (Space, f64, bool, bool) = game.step();
        let space = info.0;
        let (obs_new, action) = space;
        let reward = info.1;
        let is_terminated = info.2;
        let is_truncated = info.3;
        episode_reward += reward;
        episode_steps.push(EpisodeStep {
            observation: obs_old,
            action,
        });
        todo!()
    }
}
