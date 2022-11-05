use std::collections::HashMap;

mod bisrgraph;
mod test;
mod util;

fn main() {
    use crate::util::expected_success_distribution;
    use bisrgraph::BiSRGraph;
    let edges = vec![
        ("A", "a"),
        ("B", "a"),
        ("C", "a"),
        ("B", "b"),
        ("C", "b"),
        ("C", "c"),
    ];
    let v_weight = HashMap::from([("a", 1.), ("b", 1.1), ("c", 0.9)]);
    let g = BiSRGraph::from_edge(edges, v_weight);
    println!("{:?}", g);

    println!("{:?}", expected_success_distribution(4,1.));

    println!("{:?}", g.ALG() / 3.);

    println!("Hello, world!");
}
