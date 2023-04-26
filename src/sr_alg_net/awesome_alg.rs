use crate::sr_alg_net::util::{sample_from_softmax, transmute_act};

use super::env::{IsAdj, Load, ObservationSpace, Seq, SeqTrans};
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
pub const LABELS: usize = M;
type AlgInfo = (usize, Option<Arc<dyn Module>>);

pub static DEVICE: Lazy<Mutex<Device>> = Lazy::new(|| Device::cuda_if_available().into());

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
    pub offline_nodes_available: Vec<IsAdj>,
    pub offline_nodes_seq: Vec<Seq>,
    pub offline_nodes_loads: Vec<Prob>,
    // deep neural network
    pub policy_net: Option<Arc<dyn Module>>,
}

impl AwesomeAlg {
    pub fn get_state(&self, online_adjacent: &Vec<(usize, Prob)>) -> ObservationSpace {
        let mut load = [Load::default(); M];
        let mut seq = [SeqTrans::default(); M];
        let mut prob = [Prob::default(); M];
        let mut adj_avail = [true; M];

        let available_offline_nodes: Vec<(usize, f64)> =
            get_available_offline_nodes_in_weighted_onlineadj(
                &self.offline_nodes_available,
                online_adjacent,
            );
        let rank_vec = self
            .offline_nodes_seq
            .iter()
            .map(|&a| a as f64 / M as f64)
            .collect::<Vec<f64>>();
        let mut adj_avail_vec = vec![false; M];
        let mut prob_vec: Vec<f64> = vec![0.; M];
        for (i, p) in available_offline_nodes {
            adj_avail_vec[i] = true;
            prob_vec[i] = p;
        }
        for i in 0..M {
            load[i] = self.offline_nodes_loads[i];
            seq[i] = rank_vec[i];
            prob[i] = prob_vec[i];
            adj_avail[i] = adj_avail_vec[i]
        }
        (load, seq, prob, adj_avail)
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
        let mut offline_nodes_rank = Vec::with_capacity(l);
        for i in 0..l {
            offline_nodes_rank.push(i as i32)
        }
        offline_nodes_rank.shuffle(&mut thread_rng());
        AwesomeAlg {
            offline_nodes_available,
            offline_nodes_seq: offline_nodes_rank,
            offline_nodes_loads,
            policy_net: net,
        }
    }

    fn dispatch(self: &mut Self, online_adjacent: &Vec<(usize, Prob)>) -> Option<(usize, Prob)> {
        let obs: ObservationSpace = self.get_state(online_adjacent);
        let obs_tensor: Tensor = util::transmute_obs(obs);
        let action_raw_tensor = self.policy_net.clone().unwrap().forward(&obs_tensor);
        let action_prob = transmute_act(&action_raw_tensor).0;
        let action = sample_from_softmax(&action_prob);
        let probs = obs.2;
        let prob: f64 = probs[action];
        let is_adj = obs.3;
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
