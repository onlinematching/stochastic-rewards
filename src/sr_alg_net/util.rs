use tch::Tensor;

use super::env::ObservationSpace;

pub const M: usize = 3;

pub const fn pow2(n: usize) -> usize {
    1 << n
}

pub fn transmute_obs(obs: ObservationSpace) -> Tensor {
    let mut obs_transmute: Vec<f64> = Vec::new();
    obs_transmute.extend(obs.0);
    obs_transmute.extend(obs.1);
    obs_transmute.extend(obs.2);
    obs_transmute.extend(obs.3.map(|a| match a {
        true => 1.,
        false => 0.,
    }));
    Tensor::of_slice(&obs_transmute)
}
