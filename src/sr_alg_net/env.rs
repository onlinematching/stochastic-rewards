use super::{
    awesome_alg::{AwesomeAlg, State},
    util::M,
};
use onlinematching::{
    papers::stochastic_reward::{
        graph::{algorithm::AdaptiveAlgorithm, Prob, StochasticReward},
        mp12::Balance,
        ranking::Ranking,
    },
    weightedbigraph::WBigraph,
};
use rand::{distributions::Uniform, Rng};
use std::sync::Arc;
use tch::nn::Module;

pub type Reward = f64;
pub type Step = usize;
pub type Key = usize;
pub type Load = Prob;
pub type IsAdj = bool;

pub type ObservationSpace = ([Load; M], [Prob; M], [IsAdj; M], Step);
pub type ActionSpace = usize;
pub type ActionProbSpace = ([Prob; M],);
pub type Space = (Option<ObservationSpace>, Option<ActionSpace>);

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
            adaptive_alg: AwesomeAlg::init((M, None, State::Train)),
            step: usize::MAX,
        }
    }

    pub fn generate_random_sr() -> StochasticReward<Key> {
        // let range = Uniform::new_inclusive(1, 10);
        // let mut rng = rand::thread_rng();
        // let m = rng.sample(range);
        let m = 20;
        onlinematching::papers::stochastic_reward::mp12::example::gk(M, m)
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

    pub fn get_alg(&self) -> f64 {
        let mut alg_sum: f64 = 0.;
        let net: Option<Arc<dyn Module>> = self.adaptive_alg.deep_q_net.clone();
        for _ in 0..PRECISION {
            let mut alg = AwesomeAlg::init((M, net.clone(), State::Infer));
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
    pub fn reset(&mut self, deep_q_net: Arc<dyn Module>, state: State, _seed: i64) -> ObservationSpace {
        let graph: StochasticReward<Key> = Self::generate_random_sr();
        // if DEBUG {
        //     println!("random_sr graph = {:?}", graph)
        // }
        self.online_graph = graph;
        self.step = Step::default();
        self.adaptive_alg = AwesomeAlg::init((M, Some(deep_q_net), state));
        let adj = self.get_online_adjacent();
        self.adaptive_alg.get_state(&adj)
    }

    pub fn step(&mut self) -> (Space, Reward, bool) {
        let alg_choose = self.adaptive_alg.dispatch(&self.get_online_adjacent());
        let success = self.adaptive_alg.query_success(alg_choose);
        self.step += 1;
        let obs: Option<ObservationSpace>;
        let last_step = self.online_graph.weighted_bigraph.v_nodes.len();
        let is_terminated = self.step == last_step;
        if is_terminated {
            obs = None;
        } else {
            obs = Some(self.adaptive_alg.get_state(&self.get_online_adjacent()));
        }
        match alg_choose {
            Some((action, _p)) => {
                let reward: f64 = match success.unwrap() {
                    true => 1.,
                    false => 0.,
                };
                ((obs, Some(action)), reward, is_terminated)
            }
            None => ((obs, None), 0., is_terminated),
        }
    }
}
