#[cfg(test)]
mod test {
    use onlinematching::papers::stochastic_reward::{mp12, ranking};

    #[test]
    fn test1() {
        println!("Hello, world!");
    }

    #[test]
    fn test2() {
        let t = 100000;
        let m = 6;
        let sr = mp12::example::gk(3, m);
        let opt = sr.OPT();
        // let alg = sr.adaptive_ALG::<mp12::Balance>(t);
        let alg = sr.adaptive_ALG::<ranking::Ranking>(t);
        let ratio = alg / opt;
        println!("opt = {:?}, alg = {:?}, ratio = {:?}", opt, alg, ratio);
    }
}

