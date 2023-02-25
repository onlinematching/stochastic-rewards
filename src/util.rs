use libm::{exp, pow};

type Index = usize;
type Weight = f64;

fn factorial(num: usize) -> usize {
    match num {
        0 => 1,
        1 => 1,
        _ => factorial(num - 1) * num,
    }
}

pub fn expected_success_distribution(n: Index, lambda: Weight) -> Vec<f64> {
    let mut distro = vec![];
    let mut s = 0.;
    let p = exp(-lambda);
    for k in 0..n {
        let pow_lambda = pow(lambda, k as f64);
        let fac_k = factorial(k) as f64;
        let t = p * pow_lambda / fac_k;
        s += t;
        distro.push(t);
    }
    distro.push(1. - s);
    distro
}

