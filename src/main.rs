#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unreachable_code)]

use anyhow::Result;
use once_cell::sync::Lazy;
use play::play::iterate_batches;
use policy::policy::PolicyNet;
use std::collections::HashMap;
use std::sync::Mutex;
use tch::nn;
use tch::nn::{Module, OptimizerConfig};
use tch::Device;
use tch::Tensor;
mod bisrgraph;
mod env;
mod play;
mod policy;
mod test_edge;
mod test_net;
mod util;

static DEVICE: Lazy<Mutex<Device>> = Lazy::new(|| Device::cuda_if_available().into());

pub fn run() -> Result<()> {
    let mut env = env::env::BiSRGraphGame::new();
    let vs = nn::VarStore::new(*DEVICE.lock().unwrap());
    let vs_ref_binding = vs.root();
    let net = PolicyNet::new(&vs_ref_binding);
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    for epoch in 1..1000 {
        let batch = iterate_batches(&mut env, &net, 128);
        
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
