use onlinematching::papers::stochastic_reward::{mp12, ranking::Ranking};

use crate::sr_alg_net::util::{from_nonweight_edges, generate_worst_edges};

pub fn unitest_balance_graph(m: usize) -> f64 {
    let t = 100000;
    let edges = generate_worst_edges();
    let g = from_nonweight_edges(&edges, m);
    println!("{:?}", g);

    let sr = g.into_stochastic_reward();
    let opt = sr.OPT();
    let alg = sr.adaptive_ALG::<mp12::Balance>(t);
    let ratio = alg / opt;
    println!("opt = {:?}, alg = {:?}, ratio = {:?}", opt, alg, ratio);
    return ratio
}
