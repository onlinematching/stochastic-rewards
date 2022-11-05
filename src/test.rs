#[cfg(test)]
mod tests {
    use crate::bisrgraph::BiSRGraph;
    use crate::util::expected_success_distribution;

    #[test]
    fn test1() {
        let edges = vec![
            ("A", "a"),
            ("B", "a"),
            ("C", "a"),
            ("D", "a"),
            ("A", "b"),
            ("B", "b"),
            ("C", "b"),
            ("D", "b"),
            ("C", "c"),
            ("D", "c"),
            ("C", "d"),
            ("D", "d"),
        ];
        let v_weight = vec![("a", 0.5), ("b", 0.5), ("c", 0.5), ("d", 0.5)];
        let g = BiSRGraph::from_edge(edges, v_weight);
        println!("{:?}", g);

        println!("{:?}", g.ALG() / 2.);

        println!("Hello, world!");
    }

    #[test]
    fn test2() {
        let edges = vec![
            ("A", "a"),
            ("B", "a"),
            ("A", "b"),
            ("B", "b"),
        ];
        let v_weight = vec![("a", 0.5), ("b", 0.5)];
        let g = BiSRGraph::from_edge(edges, v_weight);
        println!("{:?}", g);

        println!("{:?}", g.ALG() / 1.);

        println!("Hello, world!");
    }

    #[test]
    fn test_util() {
        println!("{:?}", expected_success_distribution(2,1.));
        println!("Hello, world!");
    }
}
