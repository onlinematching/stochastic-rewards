use crate::sr_alg_net::util::{sample_from_softmax, tensor2actprob};

use super::env::{IsAdj, Load, ObservationSpace};
use super::util;
use once_cell::sync::Lazy;
use onlinematching::papers::adwords::util::get_available_offline_nodes_in_weighted_onlineadj;
use onlinematching::papers::stochastic_reward::graph::algorithm::AdaptiveAlgorithm;
use onlinematching::papers::stochastic_reward::graph::Prob;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::sync::{Arc, Mutex};
use tch::nn::Module;
use tch::Device;
use tch::{nn, Tensor};

const M: usize = util::M;
type AlgInfo = (usize, Option<Arc<dyn Module>>);

pub static DEVICE: Lazy<Mutex<Device>> = Lazy::new(|| Device::cuda_if_available().into());

pub fn deep_q_net(vs: &nn::Path) -> impl Module {
    const HIDDEN_LAYER1: i64 = util::pow2(M + 3) as i64;
    const HIDDEN_LAYER2: i64 = util::pow2(M + 2) as i64;
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            // Observation Spave dim + Action Space dim
            (3 * M + M) as i64,
            HIDDEN_LAYER1,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "layer2",
            HIDDEN_LAYER1,
            HIDDEN_LAYER2 as i64,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "layer2",
            HIDDEN_LAYER2,
            1 as i64,
            Default::default(),
        ))
}

#[derive(Debug)]
pub struct AwesomeAlg {
    pub offline_nodes_available: Vec<IsAdj>,
    pub offline_nodes_loads: Vec<Prob>,
    // deep Q network
    pub deep_q_net: Option<Arc<dyn Module>>,
}

impl AwesomeAlg {
    pub fn get_state(&self, online_adjacent: &Vec<(usize, Prob)>) -> ObservationSpace {
        let mut load = [Load::default(); M];
        let mut prob = [Prob::default(); M];
        let mut adj_avail = [true; M];

        let available_offline_nodes: Vec<(usize, f64)> =
            get_available_offline_nodes_in_weighted_onlineadj(
                &self.offline_nodes_available,
                online_adjacent,
            );
        let mut adj_avail_vec = vec![false; M];
        let mut prob_vec: Vec<f64> = vec![0.; M];
        for (i, p) in available_offline_nodes {
            adj_avail_vec[i] = true;
            prob_vec[i] = p;
        }
        for i in 0..M {
            load[i] = self.offline_nodes_loads[i];
            prob[i] = prob_vec[i];
            adj_avail[i] = adj_avail_vec[i]
        }
        (load, prob, adj_avail)
    }
}

impl AdaptiveAlgorithm<(usize, Prob), AlgInfo> for AwesomeAlg {
    fn init(info: AlgInfo) -> Self {
        let (l, net) = info;
        assert_eq!(
            l, M,
            "This AdaptiveAlgorithm now only available for hyperparameter M length U"
        );
        let mut offline_nodes_available = Vec::with_capacity(l);
        offline_nodes_available.resize(l, true);
        let mut offline_nodes_loads: Vec<Prob> = Vec::with_capacity(l);
        offline_nodes_loads.resize(l, 0.);
        AwesomeAlg {
            offline_nodes_available,
            offline_nodes_loads,
            deep_q_net: net,
        }
    }

    fn dispatch(self: &mut Self, online_adjacent: &Vec<(usize, Prob)>) -> Option<(usize, Prob)> {
        let obs: ObservationSpace = self.get_state(online_adjacent);
        let obs_tensor: Tensor = util::obser2tensor(obs);
        let action_raw_tensor = self.deep_q_net.clone().unwrap().forward(&obs_tensor);
        let action_prob = tensor2actprob(&action_raw_tensor).0;
        let action = sample_from_softmax(&action_prob);
        let probs = obs.1;
        let prob = probs[action];
        let is_adj = obs.2;
        if is_adj[action] {
            Some((action, prob))
        } else {
            None
        }
    }

    fn query_success(self: &mut Self, offline_node: Option<(usize, Prob)>) -> Option<bool> {
        match offline_node {
            Some(adj_info) => {
                let mut rng = rand::thread_rng();
                let prob = adj_info.1;
                let result = rng.gen_bool(prob);
                if result {
                    self.offline_nodes_available[adj_info.0] = false;
                }
                Some(result)
            }
            None => None,
        }
    }

    fn alg_output(self: Self) -> f64 {
        self.offline_nodes_available
            .iter()
            .map(|&avail| match avail {
                true => 0,
                false => 1,
            })
            .sum::<i32>() as f64
    }
}
