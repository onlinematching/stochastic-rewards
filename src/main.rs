mod graph_reinforce;
mod sr_graph_net;

fn main() {
    println!("{:?}", *graph_reinforce::DEVICE.lock().unwrap());
    match graph_reinforce::run() {
        Ok(_) => {}
        Err(e) => {
            panic!("{:?}", e)
        }
    };
}
