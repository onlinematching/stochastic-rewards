mod bisrgraph;
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
    let v_weight = vec![("a", 1.), ("b", 0.9), ("c", 0.95)];
    let g = BiSRGraph::from_edge(edges, v_weight);
    println!("{:?}", g);

    println!("{:?}", expected_success_distribution(15,1.));

    println!("Hello, world!");
}
