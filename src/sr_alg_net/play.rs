use std::sync::Arc;

use crate::sr_alg_net::env::Space;
use tch::nn::Module;

use super::env::{ActionSpace, AdapticeAlgGame, ObservationSpace};

type Reward = f64;

const SEED: i64 = 42;

#[derive(Clone, Debug)]
pub struct EpisodeStep {
    observation: ObservationSpace,
    action: Option<ActionSpace>,
}

#[derive(Debug)]
pub struct Episode {
    reward: Reward,
    steps: Vec<EpisodeStep>,
}

