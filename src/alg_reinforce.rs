use anyhow::{Result, Ok};
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;
use tch::nn;
use tch::nn::{Module, OptimizerConfig};
use tch::Device;
use tch::Tensor;

pub static DEVICE: Lazy<Mutex<Device>> = Lazy::new(|| Device::cuda_if_available().into());

const PERCENTILE: f64 = 0.3;

pub fn run() -> Result<()> {
    Ok(())
}
