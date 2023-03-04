#![allow(dead_code)]

pub mod policy {
    use tch::nn::Module;
    use tch::{nn, Tensor};

    use crate::env::env::{ActionProbabilitySpace, ObservationSpace, A};
    use crate::util::sigmoid;

    pub const M: usize = crate::util::M;

    pub fn policy_net(vs: &nn::Path) -> impl Module {
        const HIDDEN_LAYER: i64 = 128;
        let innernet = nn::seq()
            .add(nn::linear(
                vs / "layer1",
                (M * M + 1) as i64,
                HIDDEN_LAYER,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs, HIDDEN_LAYER, M as i64, Default::default()));
        innernet
    }

    pub fn transmute_observation(obs: &ObservationSpace) -> Tensor {
        let mut obs_vec = vec![];
        obs_vec.push(obs.1 as f64);
        let obs = &obs.0;
        for v_adj in obs {
            for u in v_adj {
                if u == &A::Success {
                    obs_vec.push(1.)
                } else {
                    obs_vec.push(0.)
                }
            }
        }
        Tensor::of_slice(&obs_vec)
    }

    pub fn transmute_action(raw_action: &Tensor) -> ActionProbabilitySpace {
        let action = Vec::<f64>::from(raw_action.view(-1));
        let mut action: ActionProbabilitySpace = action
            .as_slice()
            .try_into()
            .expect("slice with incorrect length");
        for i in 0..M {
            action[i] = sigmoid(action[i]);
        }
        action
    }
}
