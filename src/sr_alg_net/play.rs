use std::sync::Arc;

use crate::sr_alg_net::env::Space;
use tch::nn::Module;

use super::env::{ActionSpace, AdapticeAlgGame, ObservationSpace};

type Reward = f64;

const SEED: i64 = 42;
const LEARNING_RATE: f64 = 0.98;

#[derive(Clone, Debug)]
pub struct EpisodeStep {
    observation: ObservationSpace,
    action: Option<ActionSpace>,
}

#[derive(Debug)]
pub struct Episode {
    reward: Reward,
    steps: Vec<EpisodeStep>,
}

pub fn play_step(game: &mut AdapticeAlgGame, deep_q_net: Arc<dyn Module>) {
    let mut total_reward: Reward = Reward::default();
    let mut state: ObservationSpace = game.reset(deep_q_net.clone(), SEED);

}

pub fn calculate_loss() {}
