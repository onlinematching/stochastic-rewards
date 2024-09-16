use crate::sr_alg_net::util::pow2;
use sr_alg_net::uni_test::{self, unitest_balance_graph};
use sr_graph_net::test_edge::uni_test::uni_test_opt;
use crate::sr_alg_net::util::{from_nonweight_edges, generate_worst_edges, M};
mod alg_reinforce;
mod graph_reinforce;
mod sr_alg_net;
mod sr_graph_net;

fn main() {
    println!("{:?}", *graph_reinforce::DEVICE.lock().unwrap());
    let n = 7;
    let m = pow2(n);
    let r = uni_test::unitest_balance_graph_b(m);
    println!("{:?}", r);
    match alg_reinforce::run(16) {
        Ok(_) => {}
        Err(e) => {
            panic!("{:?}", e)
        }
    };
    match graph_reinforce::run() {
        Ok(_) => {}
        Err(e) => {
            panic!("{:?}", e)
        }
    };
}
