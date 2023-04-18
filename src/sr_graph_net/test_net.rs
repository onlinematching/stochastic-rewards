#[cfg(test)]
mod tests_net {
    use std::collections::HashMap;

    use crate::sr_graph_net::{bisrgraph, util::expected_success_distribution};

    #[test]
    fn test1() {
        println!("Hello, world!");
    }

    #[test]
    fn test_util_expected_success_distribution() {
        println!("{:?}", expected_success_distribution(2, 1.));
        println!("Hello, world!");
    }

    #[test]
    fn test_main() {
        use crate::sr_graph_net::util::expected_success_distribution;
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

        println!("{:?}", expected_success_distribution(4, 1.));

        println!("{:?}", g.ALG() / 3.);

        println!("Hello, world!");
    }
}
