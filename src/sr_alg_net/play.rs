use crate::sr_alg_net::util::deep_q_net_pretransmute;
use crate::sr_alg_net::{awesome_alg::get_best_action_and_reward, env::Space};
use std::sync::Arc;
use tch::Reduction;
use tch::{nn::Module, Tensor};

use super::awesome_alg::State;
use super::env::{ActionSpace, AdapticeAlgGame, ObservationSpace, DEBUG};

type Reward = f64;

const SEED: i64 = 42;
const GAMMA: f32 = 1.;

#[derive(Clone, Debug, Copy)]
pub struct Experience {
    pub state: ObservationSpace,
    pub action: Option<ActionSpace>,
    pub reward: Reward,
    // if new_state == None which means the Agent has been terminated.
    // So no need for done flag
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
    pub fn new() -> Self {
        ExperienceBuffer { buffer: Vec::new() }
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

pub fn play(
    game: &mut AdapticeAlgGame,
    deep_q_net: Arc<dyn Module>,
    state: State,
) -> Option<ExperienceBuffer> {
    if state == State::Infer {
        println!("alg = {}", game.get_alg())
    }
    let mut buffer = ExperienceBuffer::new();
    let mut state: ObservationSpace = game.reset(deep_q_net.clone(), state, SEED);
    loop {
        let info: (Space, f64, bool) = game.step();
        let (space, reward, is_terminated) = info;
        let (new_state, action) = space;
        if DEBUG {
            match new_state {
                Some(_) => assert!(!is_terminated),
                None => assert!(is_terminated),
            }
        }
        let exp = Experience {
            state,
            action,
            reward,
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
    net: Arc<dyn Module>,
) -> Tensor {
    let state_action = Tensor::stack(
        &state_action
            .into_iter()
            .map(deep_q_net_pretransmute)
            .collect::<Vec<Tensor>>(),
        0,
    );
    let state_action_v = net.forward(&state_action);
    // if DEBUG {
    //     println!("state_action = ");
    //     state_action.print();
    //     println!("state_action_v = ");
    //     state_action_v.print();
    // }
    // return is discount reward: reward + Gamma * max_Q(s_{n+1}, a)
    let expected_state_action_v = tch::no_grad(|| {
        let rewards = Tensor::of_slice(&reward.iter().map(|&r| r as f32).collect::<Vec<f32>>());
        // rewards.print();
        let next_states_q_net_return = Tensor::of_slice(
            &next_states
                .into_iter()
                .map(|o| {
                    o.map(|obs: ObservationSpace| get_best_action_and_reward(obs, net.clone()).1)
                })
                .map(|o| match o {
                    Some(r) => r as f32,
                    None => 0.,
                })
                .collect::<Vec<f32>>(),
        );
        // next_states_q_net_return.print();
        rewards + GAMMA * next_states_q_net_return
    });
    // println!("expected_state_action_v = ");
    // expected_state_action_v.print();
    state_action_v.mse_loss(&expected_state_action_v, Reduction::Mean)
}
