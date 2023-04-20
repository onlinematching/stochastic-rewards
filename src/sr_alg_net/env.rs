use super::util::M;
use onlinematching::papers::stochastic_reward::graph::{Prob, StochasticReward};

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
pub type IsAdj = bool;

pub type ObservationSpace = ([Load; M], [Rank; M], [Prob; M], [IsAdj; M]);
pub type ActionSpace = usize;

pub const ALPHA: f64 = 0.5;

pub struct AdapticeAlgGame {
    online_graph: StochasticReward<Key>,
    adaptive_alg: super::awesome_alg::AwesomeAlg,
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

    fn get_state(&self) -> ObservationSpace {
        
        todo!()
    }
}

impl Env for AdapticeAlgGame {
    fn reset(&mut self, _seed: i64) -> (ObservationSpace, Reward, bool, bool) {
        let graph = Self::generate_random_sr();

        (self.get_state(), 0., false, false)
    }

    fn step(&mut self, action: &ActionSpace) -> (ObservationSpace, Reward, bool, bool) {
        todo!()
    }
}
