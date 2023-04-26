use super::{
    awesome_alg::AwesomeAlg,
    util::{bernoulli_trial, M},
};
use ndarray_rand::rand_distr::Uniform;
use onlinematching::{
    papers::stochastic_reward::{
        graph::{algorithm::AdaptiveAlgorithm, Prob, StochasticReward},
        mp12::Balance,
        ranking::Ranking,
    },
    weightedbigraph::WBigraph,
};
use rand::Rng;
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
pub type Space = (ObservationSpace, Option<ActionSpace>);

pub const DEBUG: bool = true;
pub const PRECISION: usize = 1000;

pub struct AdapticeAlgGame {
    pub online_graph: StochasticReward<Key>,
    pub adaptive_alg: super::awesome_alg::AwesomeAlg,
    pub step: usize,
}

impl AdapticeAlgGame {
    pub fn new() -> AdapticeAlgGame {
        Self {
            online_graph: WBigraph::new().into_stochastic_reward(),
            adaptive_alg: AwesomeAlg::init((M, None)),
            step: usize::MAX,
        }
    }

    pub fn generate_random_sr() -> StochasticReward<Key> {
        let range = Uniform::new_inclusive(1, 10);
        let mut rng = rand::thread_rng();
        let m = rng.sample(range);
        let gk = onlinematching::papers::stochastic_reward::mp12::example::gk(M, m);
        println!("gk = {:?}", gk);
        gk
    }

    fn get_online_adjacent(&self) -> Vec<(usize, Prob)> {
        self.online_graph.weighted_bigraph.v_adjacency_list[self.step].clone()
    }

    fn normal_alg_geometric_mean(&self, precision: usize) -> f64 {
        let alg_ranking = self.online_graph.adaptive_ALG::<Ranking>(precision);
        let alg_balance = self.online_graph.adaptive_ALG::<Balance>(precision);
        if DEBUG {
            println!("alg_ranking = {alg_ranking}, alg_balance = {alg_balance}");
        }
        (alg_ranking * alg_balance).sqrt()
    }

    fn get_alg(&self) -> f64 {
        let mut alg_sum: f64 = 0.;
        let net: Option<Arc<dyn Module>> = self.adaptive_alg.policy_net.clone();
        for _ in 0..PRECISION {
            let mut alg = AwesomeAlg::init((M, net.clone()));
            for online_adj in self.online_graph.iter() {
                let alg_choose = alg.dispatch(online_adj);
                alg.query_success(alg_choose);
            }

            alg_sum += alg.alg_output();
        }
        alg_sum / PRECISION as f64
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
        let obs: ObservationSpace = self.adaptive_alg.get_state(&online_adj);
        self.adaptive_alg.query_success(alg_choose);
        self.step += 1;
        match alg_choose {
            Some((action, _prob)) => {
                if self.step == M {
                    let ratio_contrast = self.normal_alg_geometric_mean(PRECISION) / M as f64;
                    let true_ratio = self.get_alg() / M as f64;
                    let reward = true_ratio / ratio_contrast;
                    if DEBUG {
                        println!("ratio_contrast = {ratio_contrast}, true_ratio = {true_ratio}, reward = {reward}");
                    }
                    ((obs, Some(action)), reward, true, false)
                } else {
                    ((obs, Some(action)), 0., false, false)
                }
            }
            None => ((obs, None), 0., false, true),
        }
    }
}
