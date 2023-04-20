use super::util;

const M: usize = util::M;

pub struct ActionSpace {
    value: usize,
}

pub enum A {
    Success,
    Unsuccess,
}

pub type Reward = f64;
pub type Step = usize;
pub type LoadSpace = [f64; M];
pub type RankSpace = [usize; M];
pub type AvailSpace = [A; M];

pub type ObservationSpace = (LoadSpace, RankSpace, AvailSpace);

impl ActionSpace {
    pub fn new(value: usize) -> Result<Self, String> {
        if value < M {
            Ok(ActionSpace { value })
        } else {
            Err(format!("Value should be in range 0 to {}", M))
        }
    }

    pub fn get(&self) -> usize {
        self.value
    }
}



pub trait Env {
    fn reset(&mut self, seed: i64) -> (ObservationSpace, Reward, bool, bool);

    // step(action) -> next_obs, reward, is_terminated, is_truncated
    fn step(&mut self, action: &ActionSpace) -> (ObservationSpace, Reward, bool, bool);
}

