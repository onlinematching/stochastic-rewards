use super::util;
use onlinematching::papers::stochastic_reward::graph::algorithm::AdaptiveAlgorithm;
use onlinematching::papers::stochastic_reward::graph::{OfflineInfo, Prob};
use tch::nn::Module;
use tch::{nn, Tensor};

#[derive(Debug)]
pub struct AwesomeAlg {
    offline_nodes_available: Vec<bool>,
    offline_nodes_rank: Vec<i32>,
    offline_nodes_loads: Vec<Prob>,
}

impl AdaptiveAlgorithm<(usize, Prob), OfflineInfo> for AwesomeAlg {
    fn init(lenth: OfflineInfo) -> Self {
        todo!()
    }

    fn dispatch(self: &mut Self, online_adjacent: &Vec<(usize, Prob)>) -> Option<(usize, Prob)> {
        todo!()
    }

    fn query_success(self: &mut Self, offline_node: Option<(usize, Prob)>) -> Option<bool> {
        todo!()
    }

    fn alg_output(self: Self) -> f64 {
        todo!()
    }
}
