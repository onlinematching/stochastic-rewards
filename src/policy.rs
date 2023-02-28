pub mod policy {
    use tch::nn;
    use tch::nn::Module;

    const M: i64 = 5;

    const HIDDEN_LAYER: i64 = 125;

    pub fn policy_net(vs: &nn::Path) -> impl Module {
        nn::seq()
            .add(nn::linear(
                vs / "layer1",
                M * M,
                HIDDEN_LAYER,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs, HIDDEN_LAYER, M, Default::default()))
    }
}
