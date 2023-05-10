#[cfg(test)]
mod test {
    use onlinematching::{weightedbigraph::WBigraph, papers::stochastic_reward::mp12};

    use crate::sr_alg_net::util::{M, generate_worst_g};

    #[test]
    fn test1() {
        println!("Hello, world!");
    }


    #[test]
    fn test_balance_graph() {
        let t = 10000;
        let g = generate_worst_g();
        let sr = g.into_stochastic_reward();
        let opt = sr.OPT();
        let alg = sr.adaptive_ALG::<mp12::Balance>(t);
        let ratio = alg / opt;
        println!("opt = {:?}, alg = {:?}, ratio = {:?}", opt, alg, ratio);
    }


}

