use std::collections::HashMap;

use libm::{exp, pow};

type Index = usize;
type Weight = f64;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

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

pub fn array2weight<T>(v: Vec<T>) -> HashMap<T, f64>
where
    T: std::hash::Hash + std::cmp::Eq,
{
    let mut map = HashMap::new();
    for k in v.into_iter() {
        map.insert(k, 1.);
    }
    map
}


