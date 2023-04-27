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

pub fn iterate_batches(
    game: &mut AdapticeAlgGame,
    deep_q_net: Arc<dyn Module>,
    batch_size: usize,
) -> Vec<Episode> {
    let mut batch: Vec<Episode> = vec![];
    let mut episode_reward: f64 = 0.;
    let mut episode_steps: Vec<EpisodeStep> = vec![];
    let mut obs_old: ObservationSpace = game.reset(deep_q_net.clone(), SEED);
    loop {
        let info: (Space, f64, bool, bool) = game.step();
        let space: Space = info.0;
        let (mut obs_new, action) = space;
        let reward = info.1;
        let is_terminated = info.2;
        let is_truncated = info.3;
        episode_reward += reward;
        episode_steps.push(EpisodeStep {
            observation: obs_old,
            action,
        });
        if is_terminated || is_truncated {
            let episode: Episode = Episode {
                reward: episode_reward,
                steps: episode_steps.clone(),
            };
            if reward > 0. || !is_truncated {
                batch.push(episode);
            };
            episode_reward = 0.;
            episode_steps.clear();
            obs_new = game.reset(deep_q_net.clone(), 42);
            if batch.len() == batch_size {
                return batch;
            }
        }
        obs_old = obs_new;
    }
}

pub fn filter_batch(
    batch: Vec<Episode>,
    percentile: f64,
) -> (
    Vec<ObservationSpace>,
    Vec<ActionSpace>,
    Reward,
    Reward,
    Reward,
) {
    assert!(0. <= percentile && percentile <= 100.);
    let rewards = batch
        .iter()
        .map(|episode| episode.reward)
        .collect::<Vec<f64>>();
    let reward_mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let reward_lowest = *rewards
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let reward_bound = crate::sr_graph_net::util::percentile(rewards, percentile);

    let mut train_obs: Vec<ObservationSpace> = Vec::new();
    let mut train_act: Vec<ActionSpace> = Vec::new();
    for Episode { reward, ref steps } in batch {
        if reward > reward_bound {
            continue;
        }
        train_obs.extend(steps.iter().map(|step| step.observation));
        train_act.extend(steps.iter().map(|step| step.action.unwrap()));
    }
    return (
        train_obs,
        train_act,
        reward_bound,
        reward_mean,
        reward_lowest,
    );
}
