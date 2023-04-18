pub mod env {
    use crate::sr_graph_net::{
        policy::{self, policy::pow2},
        util::{agent_generate_graph, check_symmetry_property},
    };

    const M: usize = crate::sr_graph_net::util::M;

    #[derive(PartialEq, Clone, Copy, Debug)]
    pub enum A {
        Success,
        Fail,
    }

    const LABELS: usize = pow2(M);
    pub type ActionSpace = [A; M];
    pub type ObservationSpace = ([[A; M]; M], usize);
    pub type ActionProbabilitySpace = [f32; LABELS];

    pub type Reward = f64;

    pub trait Env {
        fn reset(&mut self, seed: i64) -> (ObservationSpace, Reward, bool, bool);

        // step(action) -> next_obs, reward, is_terminated, is_truncated
        fn step(&mut self, action: &ActionSpace) -> (ObservationSpace, Reward, bool, bool);
    }

    pub struct BiSRGraphGame {
        agent_state: [[A; M]; M],
        step: usize,
    }

    impl BiSRGraphGame {
        pub fn new() -> Self {
            Self {
                agent_state: [[A::Fail; M]; M],
                step: 0,
            }
        }
    }

    impl Env for BiSRGraphGame {
        fn reset(&mut self, _seed: i64) -> (ObservationSpace, Reward, bool, bool) {
            self.agent_state = [[A::Fail; M]; M];
            self.step = 0;
            ((self.agent_state, self.step), 0., false, false)
        }

        // step(action) -> next_obs, reward, is_terminated, is_truncated
        fn step(&mut self, action: &ActionSpace) -> (ObservationSpace, Reward, bool, bool) {
            self.agent_state[self.step] = action.clone();
            self.step += 1;

            if !check_symmetry_property(&self.agent_state, self.step) {
                return ((self.agent_state, self.step), 0., false, true);
            }

            if self.step == M {
                let graph = agent_generate_graph(&(self.agent_state, self.step));
                let opt = graph.OPT();
                let alg = graph.ALG();
                let ratio = alg / opt;
                return ((self.agent_state, self.step), ratio, true, false);
            }
            ((self.agent_state, self.step), 0., false, false)
        }
    }
}
