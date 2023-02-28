mod play {
    use tch::nn::Module;
    type Env = ();

    pub fn iterate_batches(env: Env, net: &dyn Module, batch_size: i64) -> Vec<i64> {
        let mut batch = vec![];
        batch
    }

    pub fn filter_batch(batch: Vec<i64>, percentile: i32) {

    }
}