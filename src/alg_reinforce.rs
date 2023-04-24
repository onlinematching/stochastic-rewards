use anyhow::{Result, Ok};
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use tch::nn;
use tch::nn::{Module, OptimizerConfig};
use tch::Device;
use tch::Tensor;
use crate::sr_alg_net::env::{Env, AdapticeAlgGame};

const PERCENTILE: f64 = 0.3;

pub fn run() -> Result<()> {
    let mut env: AdapticeAlgGame = AdapticeAlgGame::new();
    drop(env);
    Ok(())
}
