#[cfg(test)]
pub mod test {
    use onlinematching::{papers::stochastic_reward::mp12, weightedbigraph::WBigraph};

    use crate::sr_alg_net::uni_test;
    use crate::sr_alg_net::util::{from_nonweight_edges, generate_worst_edges, pow2, M};

    #[test]
    fn test1() {
        println!("Hello, world!");
    }

    #[test]
    fn test_balance_graph() {
        let mut rs = vec![];
        let n = 7;
        for i in 0..n {
            let m = pow2(i);
            let r = uni_test::unitest_unequal_balance_graph(m);
            rs.push(r)
        }
        let s: f64 = rs.iter().sum();
        println!("{:?}, mean = {:?}", rs, s / n as f64)
    }

    #[test]
    pub fn test_my_graph() {
        let n = 6;
        let m = pow2(n);
        let r = uni_test::unitest_balance_graph(m);
        println!("{:?}", r);
    }
}
