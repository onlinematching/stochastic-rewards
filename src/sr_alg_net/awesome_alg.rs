use super::env::{ActionSpace, IsAdj, Load, ObservationSpace, Reward, Space};
use super::util::{self, bernoulli_trial, deep_q_net_pretransmute, sample};
use once_cell::sync::Lazy;
use onlinematching::papers::adwords::util::get_available_offline_nodes_in_weighted_onlineadj;
use onlinematching::papers::stochastic_reward::graph::algorithm::AdaptiveAlgorithm;
use onlinematching::papers::stochastic_reward::graph::Prob;
use rand::Rng;
use std::sync::{Arc, Mutex};
use tch::nn::Module;
use tch::Device;
use tch::{nn, Tensor};

const M: usize = util::M;
const ALPHA: f64 = 0.8;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum State {
    Train,
    Infer,
}

type AlgInfo = (usize, Option<Arc<dyn Module>>, State);

pub static DEVICE: Lazy<Mutex<Device>> = Lazy::new(|| Device::cuda_if_available().into());

pub fn deep_q_net(vs: &nn::Path) -> impl Module {
    const HIDDEN_LAYER1: i64 = util::pow2(M + 3) as i64;
    const HIDDEN_LAYER2: i64 = util::pow2(M + 2) as i64;
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            // Observation Spave dim + Action Space dim
            (3 * M + 1) as i64,
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
    // distinguish train or inferance
    pub state: State,
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

pub fn get_best_action_and_reward(
    obs: ObservationSpace,
    deep_q_net: Arc<dyn Module>,
) -> (ActionSpace, Reward) {
    let actions = (0..M).collect::<Vec<ActionSpace>>();
    let mut reward: Reward;
    let spaces = actions
        .clone()
        .into_iter()
        .map(|act: ActionSpace| (Some(obs.clone()), Some(act)))
        .map(|space: Space| deep_q_net_pretransmute(space))
        .collect::<Vec<Tensor>>();
    let rewards: Vec<Reward> = spaces
        .iter()
        .map(|space_tensor| deep_q_net.forward(&space_tensor))
        .map(|reward_tensor| Vec::<f32>::from(reward_tensor.view(-1))[0] as Reward)
        .collect::<Vec<Reward>>();
    let mut action = ActionSpace::MAX;
    reward = Reward::MIN;
    for i in actions.clone().into_iter() {
        if rewards[i] > reward {
            reward = rewards[i];
            action = i;
        }
    }
    return (action, reward);
}

impl AdaptiveAlgorithm<(usize, Prob), AlgInfo> for AwesomeAlg {
    fn init(info: AlgInfo) -> Self {
        let (l, net, state) = info;
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
            state,
        }
    }

    fn dispatch(self: &mut Self, online_adjacent: &Vec<(usize, Prob)>) -> Option<(usize, Prob)> {
        let obs: ObservationSpace = self.get_state(online_adjacent);
        let actions = (0..M).collect::<Vec<ActionSpace>>();
        let action: usize;
        if self.state == State::Train && !bernoulli_trial(ALPHA) {
            action = *sample(&actions);
        } else {
            action = get_best_action_and_reward(obs, self.deep_q_net.clone().unwrap()).0;
        }
        let probs = obs.1;
        let prob = probs[action];
        let is_adj = obs.2;
        if is_adj[action] {
            self.offline_nodes_loads[action] += prob;
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
