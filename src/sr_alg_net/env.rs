use super::util;
use onlinematching::papers::stochastic_reward::graph::{Prob, StochasticReward};
use rand::{seq::SliceRandom, thread_rng};

#[derive(Copy, Clone)]
pub enum Available {
    Success,
    Unsuccess,
}

pub type Reward = f64;
pub type Step = usize;
pub type Key = usize;
pub type Load = Prob;
pub type Rank = i32;
pub type AdjSpace = bool;

pub type ObservationSpace = ();
pub type ActionSpace = usize;

pub const ALPHA: f64 = 0.5;

pub struct AdapticeAlgGame {
    agent_state: ObservationSpace,
    graph: onlinematching::papers::stochastic_reward::graph::StochasticReward<Key>,
}

pub trait Env {
    fn reset(&mut self, seed: i64) -> (ObservationSpace, Reward, bool, bool);

    // step(action) -> next_obs, reward, is_terminated, is_truncated
    fn step(&mut self, action: &ActionSpace) -> (ObservationSpace, Reward, bool, bool);
}

impl AdapticeAlgGame {
    fn generate_random_sr() -> StochasticReward<Key> {
        todo!()
    }
}

impl Env for AdapticeAlgGame {
    fn reset(&mut self, _seed: i64) -> (ObservationSpace, Reward, bool, bool) {
        todo!()
    }

    fn step(&mut self, action: &ActionSpace) -> (ObservationSpace, Reward, bool, bool) {
        todo!()
    }
}
