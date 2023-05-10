#[cfg(test)]
mod test {
    use onlinematching::{weightedbigraph::WBigraph, papers::stochastic_reward::mp12};

    use crate::sr_alg_net::util::{M, from_nonweight_edges, generate_worst_edges};
    use crate::sr_alg_net::uni_test;

    #[test]
    fn test1() {
        println!("Hello, world!");
    }

    #[test]
    fn test_balance_graph() {
        uni_test::unitest_balance_graph(200)
    }

}

