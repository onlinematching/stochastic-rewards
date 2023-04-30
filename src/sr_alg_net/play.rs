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
    pub buffer: Vec<Experience>,
}

impl ExperienceBuffer {
    fn new() -> Self {
        ExperienceBuffer { buffer: Vec::new() }
    }

    #[inline]
    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn push(&mut self, exp: Experience) {
        self.buffer.push(exp)
    }
}

pub fn play(game: &mut AdapticeAlgGame, deep_q_net: Arc<dyn Module>) -> Option<ExperienceBuffer> {
    let mut buffer = ExperienceBuffer::new();
    let mut state: ObservationSpace = game.reset(deep_q_net.clone(), SEED);
    loop {
        let info: (Space, f64, bool) = game.step();
        let (space, reward, is_terminated) = info;
        let (new_state, action) = space;
        let exp = Experience {
            state,
            action,
            reward,
            done: is_terminated,
            new_state,
        };
        buffer.push(exp);

        if is_terminated {
            return Some(buffer);
        }
        state = new_state;
    }
}

pub fn calculate_loss() {}
