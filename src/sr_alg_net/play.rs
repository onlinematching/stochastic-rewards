use std::sync::Arc;

use crate::sr_alg_net::env::Space;
use tch::nn::Module;

use super::env::{ActionSpace, AdapticeAlgGame, ObservationSpace};

type Reward = f64;

const SEED: i64 = 42;
const LEARNING_RATE: f64 = 0.98;

#[derive(Clone, Debug)]
pub struct Experience {
    state: ObservationSpace,
    action: Option<ActionSpace>,
    reward: Reward,
    done: bool,
    new_state: ObservationSpace,
}

#[derive(Debug)]
pub struct ExperienceBuffer {
    buffer: Vec<Experience>,
}

impl ExperienceBuffer {
    #[inline]
    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn push(&mut self, exp: Experience) {
        self.buffer.push(exp)
    }
}

pub fn play_step(game: &mut AdapticeAlgGame, deep_q_net: Arc<dyn Module>) {
    let mut state: ObservationSpace = game.reset(deep_q_net.clone(), SEED);
    let info: (Space, f64, bool, bool) = game.step();
    let space: Space = info.0;
    let (mut obs_new, action) = space;
    
}

pub fn calculate_loss() {}
