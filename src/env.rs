pub mod env {
    use crate::policy;

    const M: usize = crate::util::M;

    #[derive(PartialEq, Clone, Copy)]
    pub enum A {
        Success,
        Fail,
    }

    pub type ActionSpace = [A; M];
    pub type ObservationSpace = ([[A; M]; M], usize);
    pub type ActionProbabilitySpace = [f64; M];

    pub type Reward = f64;

    pub trait Env {
        fn reset(&mut self, seed: i64);

        // step(action) -> next_obs, reward, is_terminated, is_truncated
        fn step(&mut self, action: &ActionSpace) -> (ObservationSpace, Reward, bool, bool);
    }

    pub struct BiSRGraphGame {
        agent_state: [[A; M]; M],
        step: usize,
    }

    impl Env for BiSRGraphGame {
        fn reset(&mut self, _seed: i64) {
            self.agent_state = [[A::Fail; M]; M];
            self.step = 0;
        }

        // step(action) -> next_obs, reward, is_terminated, is_truncated
        fn step(&mut self, action: &ActionSpace) -> (ObservationSpace, Reward, bool, bool) {
            self.agent_state[self.step] = action.clone();
            self.step += 1;
            
            todo!();

            if self.step == M {
                return ((self.agent_state, self.step), 0., true, true);
            }
            ((self.agent_state, self.step), 0., false, false)
        }
    }
}
