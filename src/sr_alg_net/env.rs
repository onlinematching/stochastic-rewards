use super::{awesome_alg::AwesomeAlg, util::M};
use onlinematching::papers::{
    adwords::util::get_available_offline_nodes_in_weighted_onlineadj,
    stochastic_reward::graph::{algorithm::AdaptiveAlgorithm, Prob, StochasticReward},
};

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
pub type RankTrans = f64;
pub type IsAdj = bool;

pub type ObservationSpace = ([Load; M], [RankTrans; M], [Prob; M], [IsAdj; M]);
pub type ActionSpace = usize;
pub type ActionProbSpace = ([Prob; M],);

pub const ALPHA: f64 = 0.5;

pub struct AdapticeAlgGame {
    online_graph: StochasticReward<Key>,
    adaptive_alg: super::awesome_alg::AwesomeAlg,
    step: usize,
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

    fn get_online_adjacent(&self) -> Vec<(usize, Prob)> {
        self.online_graph.weighted_bigraph.v_adjacency_list[self.step].clone()
    }

}

impl Env for AdapticeAlgGame {
    fn reset(&mut self, _seed: i64) -> (ObservationSpace, Reward, bool, bool) {
        let graph = Self::generate_random_sr();
        self.online_graph = graph;
        self.step = 0;
        self.adaptive_alg = AwesomeAlg::init(M);
        let adj = self.get_online_adjacent();

        (self.adaptive_alg. get_state(&adj), 0., false, false)
    }

    fn step(&mut self, action: &ActionSpace) -> (ObservationSpace, Reward, bool, bool) {
        todo!()
    }
}
