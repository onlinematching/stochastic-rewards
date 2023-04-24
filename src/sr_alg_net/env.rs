use super::{awesome_alg::AwesomeAlg, util::M};
use onlinematching::{
    papers::stochastic_reward::{
        graph::{algorithm::AdaptiveAlgorithm, Prob, StochasticReward},
        mp12::Balance,
        ranking::Ranking,
    },
    weightedbigraph::WBigraph,
};
use std::sync::Arc;
use tch::nn::Module;

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
pub type Space = (ObservationSpace, ActionSpace);

pub const ALPHA: f64 = 0.5;

pub struct AdapticeAlgGame {
    pub online_graph: StochasticReward<Key>,
    pub adaptive_alg: super::awesome_alg::AwesomeAlg,
    pub step: usize,
}

impl AdapticeAlgGame {
    pub fn new() -> AdapticeAlgGame {
        Self {
            online_graph: WBigraph::new().into_stochastic_reward(),
            adaptive_alg: AwesomeAlg::init((0, None)),
            step: usize::MAX,
        }
    }

    pub fn generate_random_sr() -> StochasticReward<Key> {
        todo!()
    }

    pub fn get_online_adjacent(&self) -> Vec<(usize, Prob)> {
        self.online_graph.weighted_bigraph.v_adjacency_list[self.step].clone()
    }

    pub fn normal_alg_ratio_geometric_mean(&self, precision: usize) -> f64 {
        let ratio_ranking = self.online_graph.adaptive_ALG::<Ranking>(precision);
        let ratio_balance = self.online_graph.adaptive_ALG::<Balance>(precision);
        (ratio_ranking * ratio_balance).sqrt()
    }
}

impl AdapticeAlgGame {
    pub fn reset(&mut self, policy_net: Arc<dyn Module>, _seed: i64) -> ObservationSpace {
        let graph = Self::generate_random_sr();
        self.online_graph = graph;
        self.step = Step::default();
        self.adaptive_alg = AwesomeAlg::init((M, Some(policy_net)));
        let adj = self.get_online_adjacent();
        self.adaptive_alg.get_state(&adj)
    }

    pub fn step(&mut self) -> (Space, Reward, bool, bool) {
        let online_adj = self.get_online_adjacent();
        let alg_choose = self.adaptive_alg.dispatch(&online_adj);
        self.adaptive_alg.query_success(alg_choose);
        todo!()
    }
}
