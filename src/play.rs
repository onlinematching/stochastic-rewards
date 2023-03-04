#![allow(dead_code)]

pub mod play {
    use tch::nn::Module;

    use crate::{
        env::env::{ActionSpace, Env, ObservationSpace},
        policy::policy::{transmute_action, transmute_observation},
        util::sampling_array,
    };

    const DEBUG: bool = false;

    const SEED: i64 = 42;

    #[derive(Clone, Debug)]
    pub struct EpisodeStep {
        observation: ObservationSpace,
        action: ActionSpace,
    }

    #[derive(Debug)]
    pub struct Episode {
        reward: f64,
        steps: Vec<EpisodeStep>,
    }

    pub fn iterate_batches(env: &mut dyn Env, net: &dyn Module, batch_size: usize) -> Vec<Episode> {
        let mut batch = vec![];
        let mut episode_reward = 0.;
        let mut episode_steps = vec![];
        let mut obs = env.reset(SEED);
        loop {
            let obs_v = obs.0;
            let trans_obs_v = transmute_observation(&obs_v);
            let trans_act_probs_v = net.forward(&trans_obs_v);
            let act_probs_v = transmute_action(&trans_act_probs_v);
            let act_sample = sampling_array(&act_probs_v);
            let mut next_obs = env.step(&act_sample);
            let obs_value = next_obs.0;
            let reward = next_obs.1;
            let is_terminated = next_obs.2;
            let is_truncated = next_obs.3;
            episode_reward += reward;
            let step = EpisodeStep {
                observation: obs_value,
                action: act_sample,
            };
            episode_steps.push(step);
            if is_terminated || is_truncated {
                if DEBUG && is_terminated {
                    let graph = crate::util::agent_generate_graph(&obs_value);
                    println!("{:?}", graph);
                    println!("ALG = {:?}, OPT = {:?}", graph.ALG(), graph.OPT());
                    println!("rario = {:?}", graph.ALG() / graph.OPT());
                }
                let episode = Episode {
                    reward: episode_reward,
                    steps: episode_steps.clone(),
                };
                if reward > 0. {
                    batch.push(episode);
                };
                episode_reward = 0.;
                episode_steps.clear();
                next_obs = env.reset(42);
                if batch.len() == batch_size {
                    return batch;
                }
            }
            obs = next_obs;
        }
    }

    pub fn filter_batch(
        batch: Vec<Episode>,
        percentile: f64,
    ) -> (Vec<ObservationSpace>, Vec<ActionSpace>, f64, f64, f64) {
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
        let reward_bound = crate::util::percentile(rewards, percentile);

        let mut train_obs: Vec<ObservationSpace> = Vec::new();
        let mut train_act: Vec<ActionSpace> = Vec::new();
        for Episode { reward, ref steps } in batch {
            if reward > reward_bound {
                continue;
            }
            train_obs.extend(steps.iter().map(|step| step.observation));
            train_act.extend(steps.iter().map(|step| step.action));
        }
        return (
            train_obs,
            train_act,
            reward_bound,
            reward_mean,
            reward_lowest,
        );
    }
}
