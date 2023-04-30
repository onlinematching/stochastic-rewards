use crate::sr_alg_net::env::Space;
use crate::sr_alg_net::util::deep_q_net_pretransmute;
use std::sync::Arc;
use tch::{nn::Module, Tensor};

use super::env::{ActionSpace, AdapticeAlgGame, ObservationSpace};

type Reward = f64;

const SEED: i64 = 42;
const LEARNING_RATE: f64 = 0.98;

#[derive(Clone, Debug, Copy)]
pub struct Experience {
    pub state: ObservationSpace,
    pub action: Option<ActionSpace>,
    pub reward: Reward,
    pub done: bool,
    pub new_state: Option<ObservationSpace>,
}

impl Experience {
    pub fn get_space(&self) -> Space {
        (Some(self.state), self.action)
    }
}

#[derive(Debug, Clone)]
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

    pub fn bean<F, T>(&self, f: F) -> Vec<T>
    where
        F: Fn(&Experience) -> T,
    {
        self.buffer.iter().map(|exp| f(exp)).collect::<Vec<T>>()
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
        state = new_state.unwrap();
    }
}

pub fn calculate_loss(
    state_action: Vec<Space>,
    reward: Vec<Reward>,
    next_states: Vec<Option<ObservationSpace>>,
    done_mask: Vec<bool>,
    net: Arc<dyn Module>,
) -> Tensor {
    let state_action = Tensor::stack(
        &state_action
            .into_iter()
            .map(deep_q_net_pretransmute)
            .collect::<Vec<Tensor>>(),
        0,
    );
    println!("-----------");
    state_action.print();

    let state_action_v = net.forward(&state_action);
    println!("-----------");
    state_action_v.print();

    todo!()
}
