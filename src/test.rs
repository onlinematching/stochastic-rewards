#[cfg(test)]
mod tests {
    use std::collections::HashMap;

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
        let v_weight = HashMap::from([("a", 0.5), ("b", 0.5), ("c", 0.5), ("d", 0.5)]);
        let g = BiSRGraph::from_edge(edges, v_weight);
        println!("{:?}", g);

        println!("{:?}", g.ALG() / 2.);

        println!("Hello, world!");
    }

    #[test]
    fn test2() {
        let edges = vec![("A", "a"), ("B", "a"), ("A", "b"), ("B", "b")];
        let v_weight = HashMap::from([("a", 0.5), ("b", 0.5)]);
        let g = BiSRGraph::from_edge(edges, v_weight);
        println!("{:?}", g);

        println!("{:?}", g.ALG() / 1.);
    }

    #[test]
    fn test3() {
        let edges = vec![("A", "a"), ("A", "b")];
        let v_weight = HashMap::from([("a", 0.5), ("b", 0.5)]);
        let g = BiSRGraph::from_edge(edges, v_weight);
        println!("{:?}", g);
        println!("{:?}", g.ALG() / 1.);
    }

    #[test]
    fn test4() {
        let edges = vec![("A", "a"), ("B", "a")];
        let v_weight = HashMap::from([("a", 2.0)]);
        let g = BiSRGraph::from_edge(edges, v_weight);
        println!("{:?}", g);
        println!("{:?}", g.ALG() / 2.);
    }

    #[test]
    fn test5() {
        let edges = vec![
            ("A", "a"),
            ("B", "a"),
            ("C", "a"),
            ("D", "a"),
            ("B", "b"),
            ("C", "b"),
            ("D", "b"),
            ("C", "c"),
            ("D", "c"),
            ("D", "d"),
        ];
        let v_weight = HashMap::from([("a", 0.51), ("b", 1.00), ("c", 1.00), ("d", 1.00)]);
        let mut opt = 0.;
        for (_, o) in &v_weight {
            opt += o;
        }
        let g = BiSRGraph::from_edge(edges, v_weight);

        println!("{:?}", g);

        println!("opt = {:?}, ALG = {:?}", opt, g.ALG() / opt);
    }

    #[test]
    fn test6() {
        let n = 7;
        let mut edges = vec![];
        for v in 0..n {
            for u in v..n {
                edges.push((u, v));
            }
        }
        let mut v_weight = HashMap::new();
        for v in 0..n {
            v_weight.insert(v, 1.0);
        }
        let g = BiSRGraph::from_edge(edges, v_weight);

        println!("{:?}", g);

        println!("opt = {:?}, ALG = {:?}", n, g.ALG() / n as f64);
    }

    #[test]
    fn test7() {
        let edges = vec![
            ("A", "a"),
            ("B", "a"),
            ("C", "a"),
            ("D", "a"),
            ("E", "a"),
            ("F", "a"),
            ("G", "a"),
            ("B", "b"),
            ("C", "b"),
            ("D", "b"),
            ("C", "c"),
            ("D", "c"),
            ("D", "d"),
            ("E", "e"),
            ("F", "e"),
            ("G", "e"),
            ("F", "f"),
            ("G", "f"),
            ("G", "g"),
        ];
        let v_weight = HashMap::from([
            ("a", 1.00),
            ("b", 1.00),
            ("c", 1.00),
            ("d", 1.00),
            ("e", 1.00),
            ("f", 1.00),
            ("g", 1.00),
        ]);
        let mut opt = 0.;
        for (_, o) in &v_weight {
            opt += o;
        }
        let g = BiSRGraph::from_edge(edges, v_weight);

        println!("{:?}", g);

        println!("opt = {:?}, ALG = {:?}", opt, g.ALG() / opt);
    }


    #[test]
    fn test8() {
        let edges = vec![
            ("A", "a"),
            ("B", "a"),
            ("C", "a"),
            ("D", "a"),
            ("E", "a"),
            ("B", "b"),
            ("C", "b"),
            ("C", "c"),
            ("D", "d"),
            ("E", "d"),
            ("E", "e"),
        ];
        let v_weight = HashMap::from([
            ("a", 1.00),
            ("b", 1.00),
            ("c", 1.00),
            ("d", 1.00),
            ("e", 1.00),
        ]);
        let mut opt = 0.;
        for (_, o) in &v_weight {
            opt += o;
        }
        let g = BiSRGraph::from_edge(edges, v_weight);

        println!("{:?}", g);

        println!("opt = {:?}, ALG = {:?}", opt, g.ALG() / opt);
    }


    #[test]
    fn test_util() {
        println!("{:?}", expected_success_distribution(2, 1.));
        println!("Hello, world!");
    }
}
