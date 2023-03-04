#![allow(dead_code)]

mod play {
    use tch::nn::Module;

    use crate::{env::env::Env, policy::policy::PolicyNet};

    const SEED: i64 = 42;

    pub fn iterate_batches(env: &mut dyn Env, net: &PolicyNet, batch_size: i64) -> Vec<i64> {
        let mut batch = vec![];
        let mut episode_reward: f64 = 0.;
        let mut episode_steps: Vec<i64> = vec![];
        let obs = env.reset(SEED);
        loop {
            let obs_v = obs.0;
            let act_probs_v = net.forward(&obs_v);
            // let obs = env.step(action)
        }

        batch
    }

    pub fn filter_batch(batch: Vec<i64>, percentile: i32) {
        todo!()
    }
}
