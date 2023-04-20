use super::env::{AdjSpace, Rank};
use super::util;
use onlinematching::papers::adwords::util::get_available_offline_nodes_in_weighted_onlineadj;
use onlinematching::papers::stochastic_reward::graph::algorithm::AdaptiveAlgorithm;
use onlinematching::papers::stochastic_reward::graph::{OfflineInfo, Prob};
use rand::seq::SliceRandom;
use rand::thread_rng;
use tch::nn::Module;
use tch::{nn, Tensor};

const M: usize = util::M;
pub const LABELS: usize = M;

pub fn policy_net(vs: &nn::Path) -> impl Module {
    const HIDDEN_LAYER: i64 = util::pow2(M + 3) as i64;
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            (M + M + M + M) as i64,
            HIDDEN_LAYER,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs,
            HIDDEN_LAYER,
            LABELS as i64,
            Default::default(),
        ))
}

#[derive(Debug)]
pub struct AwesomeAlg {
    pub offline_nodes_available: Vec<AdjSpace>,
    pub offline_nodes_rank: Vec<Rank>,
    pub offline_nodes_loads: Vec<Prob>,
}

impl AdaptiveAlgorithm<(usize, Prob), OfflineInfo> for AwesomeAlg {
    fn init(l: OfflineInfo) -> Self {
        assert_eq!(
            l, M,
            "This AdaptiveAlgorithm now only available for hyperparameter M length U"
        );
        let mut offline_nodes_available = Vec::with_capacity(l);
        offline_nodes_available.resize(l, true);
        let mut offline_nodes_loads: Vec<Prob> = Vec::with_capacity(l);
        offline_nodes_loads.resize(l, 0.);
        let mut offline_nodes_rank = Vec::with_capacity(l);
        for i in 0..l {
            offline_nodes_rank.push(i as i32)
        }
        offline_nodes_rank.shuffle(&mut thread_rng());
        AwesomeAlg {
            offline_nodes_available,
            offline_nodes_rank,
            offline_nodes_loads,
        }
    }

    fn dispatch(self: &mut Self, online_adjacent: &Vec<(usize, Prob)>) -> Option<(usize, Prob)> {
        let available_offline_nodes = get_available_offline_nodes_in_weighted_onlineadj(
            &self.offline_nodes_available,
            online_adjacent,
        );

        todo!()
    }

    fn query_success(self: &mut Self, offline_node: Option<(usize, Prob)>) -> Option<bool> {
        todo!()
    }

    fn alg_output(self: Self) -> f64 {
        todo!()
    }
}
