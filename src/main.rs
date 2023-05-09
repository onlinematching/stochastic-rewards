mod alg_reinforce;
mod graph_reinforce;
mod sr_alg_net;
mod sr_graph_net;

fn main() {
    println!("{:?}", *graph_reinforce::DEVICE.lock().unwrap());
    match alg_reinforce::run(16) {
        Ok(_) => {}
        Err(e) => {
            panic!("{:?}", e)
        }
    };
    // match graph_reinforce::run() {
    //     Ok(_) => {}
    //     Err(e) => {
    //         panic!("{:?}", e)
    //     }
    // };
}
