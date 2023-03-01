pub mod env {
    use crate::policy;

    const M: usize = policy::policy::M;

    #[derive(PartialEq)]
    pub enum A {
        Success,
        Fail,
    }

    pub type ActionSpace = [A; M];
    pub type ObservationSpace = ([[A; M]; M], usize);
    pub type ActionProbabilitySpace = [f64; M];

    pub type Reward = f64;

    pub trait Env {
        fn reset(&self, seed: i64);

        // step(action) -> next_obs, reward, is_terminated, is_truncated
        fn step(&self, action: ActionSpace) -> (ObservationSpace, Reward, bool, bool);
    }

    pub struct BiSRGraphGame {}

    impl Env for BiSRGraphGame {
        fn reset(&self, _seed: i64) {
            todo!()
        }

        fn step(&self, action: ActionSpace) -> (ObservationSpace, Reward, bool, bool) {
            todo!()
        }
    }
}
