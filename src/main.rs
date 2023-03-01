use anyhow::Result;
use once_cell::sync::Lazy;
use policy::policy::PolicyNet;
use std::collections::HashMap;
use std::sync::Mutex;
use tch::nn::{Module, OptimizerConfig};
use tch::Tensor;
use tch::{nn, Device};
mod bisrgraph;
mod env;
mod play;
mod policy;
mod test_edge;
mod test_net;
mod util;

static DEVICE: Lazy<Mutex<Device>> = Lazy::new(|| Device::cuda_if_available().into());

pub fn run() -> Result<()> {
    let m = todo!();
    let vs = nn::VarStore::new(*DEVICE.lock().unwrap());
    let net = PolicyNet::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    for epoch in 1..200 {
        // let loss = net
        //     .forward(&m.train_x)
        //     .cross_entropy_for_logits(&m.train_labels);
        // opt.backward_step(&loss);
    }
    Ok(())
}

fn main() {
    println!("{:?}", *DEVICE.lock().unwrap());
    match run() {
        Ok(_) => {}
        Err(e) => {
            panic!("{:?}", e)
        }
    };
}
