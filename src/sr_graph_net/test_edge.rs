pub mod uni_test {
    use std::collections::HashMap;

    use crate::sr_graph_net::bisrgraph::BiSRGraph;

    pub fn uni_test_opt() {
        let edges = vec![
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (8, 0),
            (9, 0),
            (10, 0),
            (11, 0),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (8, 1),
            (9, 1),
            (10, 1),
            (11, 1),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 2),
            (8, 2),
            (9, 2),
            (10, 2),
            (11, 2),
            (3, 3),
            (4, 3),
            (5, 3),
            (6, 3),
            (7, 3),
            (8, 3),
            (9, 3),
            (10, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
            (8, 8),
            (9, 9),
            (10, 10),
            (11, 11),
        ];

        let v_weight = HashMap::from([
            (0, 1.00),
            (1, 1.00),
            (2, 1.00),
            (3, 1.00),
            (4, 1.00),
            (5, 1.00),
            (6, 1.00),
            (7, 1.00),
            (8, 1.00),
            (9, 1.00),
            (10, 1.00),
            (11, 1.00),
        ]);
        let mut opt = 0.;
        for (_, o) in &v_weight {
            opt += o;
        }
        let g = BiSRGraph::from_edge(edges, v_weight);

        println!("{:?}", g);

        println!("opt = {:?}, ALG = {:?}", opt, g.ALG() / opt);
    }
}

#[cfg(test)]
mod tests_edge {
    use crate::sr_graph_net::bisrgraph::BiSRGraph;
    use std::collections::HashMap;

    use super::uni_test::uni_test_opt;

    #[test]
    fn test_opt() {
        uni_test_opt();
    }

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
    fn test9() {
        let edges = vec![("A", "a"), ("B", "a"), ("C", "a"), ("B", "b"), ("C", "c")];
        let v_weight = HashMap::from([("a", 1.00), ("b", 1.00), ("c", 1.00)]);
        let mut opt = 0.;
        for (_, o) in &v_weight {
            opt += o;
        }
        let g = BiSRGraph::from_edge(edges, v_weight);

        println!("{:?}", g);

        println!("opt = {:?}, ALG = {:?}", opt, g.ALG() / opt);
    }

    #[test]
    fn test10() {
        let edges = vec![("A", "a"), ("B", "a"), ("B", "b")];
        let v_weight = HashMap::from([("a", 1.00), ("b", 1.00)]);
        let mut opt = 0.;
        for (_, o) in &v_weight {
            opt += o;
        }
        let g = BiSRGraph::from_edge(edges, v_weight);

        println!("{:?}", g);

        println!("opt = {:?}, ALG = {:?}", opt, g.ALG() / opt);
    }

    #[test]
    fn test11() {
        // BiSRGraph { u: 3, v_lambda: [1.0, 1.0, 1.0], u_adj: [[0], [0, 1], [0, 2]], v_adj: [[0, 1, 2], [1], [2]] }
        // opt = 3.0, ALG = 0.6097749953282586
        let edges = vec![("A", "a"), ("B", "a"), ("C", "a"), ("B", "b"), ("C", "c")];
        let v_weight = HashMap::from([("a", 1.00), ("b", 1.00), ("c", 1.00)]);
        let mut opt = 0.;
        for (_, o) in &v_weight {
            opt += o;
        }
        let g = BiSRGraph::from_edge(edges, v_weight);

        println!("{:?}", g);

        println!("opt = {:?}, ALG = {:?}", opt, g.ALG() / opt);
    }

    #[test]
    fn test12_5() {
        let edges = vec![
            ("A", "a"),
            ("B", "a"),
            ("C", "a"),
            ("D", "a"),
            ("B", "b"),
            ("C", "c"),
            ("D", "d"),
        ];
        let v_weight = HashMap::from([("a", 1.00), ("b", 1.00), ("c", 1.00), ("d", 1.00)]);
        let mut opt = 0.;
        for (_, o) in &v_weight {
            opt += o;
        }
        let g = BiSRGraph::from_edge(edges, v_weight);

        println!("{:?}", g);

        println!("opt = {:?}, ALG = {:?}", opt, g.ALG() / opt);
    }

    #[test]
    fn test12() {
        let edges = vec![
            ("A", "a"),
            ("B", "a"),
            ("C", "a"),
            ("D", "a"),
            ("E", "a"),
            ("B", "b"),
            ("C", "c"),
            ("D", "d"),
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
    fn test13() {
        let edges = vec![
            ("A", "a"),
            ("B", "a"),
            ("C", "a"),
            ("D", "a"),
            ("E", "a"),
            ("F", "a"),
            ("B", "b"),
            ("C", "c"),
            ("D", "d"),
            ("E", "e"),
            ("F", "f"),
        ];
        let v_weight = HashMap::from([
            ("a", 1.00),
            ("b", 1.00),
            ("c", 1.00),
            ("d", 1.00),
            ("e", 1.00),
            ("f", 1.00),
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
    fn test14() {
        // opt = 11.0, ALG = 0.5983306796608542

        let edges = vec![
            ("X", "0"),
            ("A0", "0"),
            ("B0", "0"),
            ("C0", "0"),
            ("D0", "0"),
            ("E0", "0"),
            ("A1", "0"),
            ("B1", "0"),
            ("C1", "0"),
            ("D1", "0"),
            ("E1", "0"),
            ("A0", "a0"),
            ("B0", "a0"),
            ("C0", "a0"),
            ("D0", "a0"),
            ("E0", "a0"),
            ("B0", "b0"),
            ("C0", "c0"),
            ("D0", "d0"),
            ("E0", "e0"),
            ("A1", "a1"),
            ("B1", "a1"),
            ("C1", "a1"),
            ("D1", "a1"),
            ("E1", "a1"),
            ("B1", "b1"),
            ("C1", "c1"),
            ("D1", "d1"),
            ("E1", "e1"),
        ];
        let v_weight = HashMap::from([
            ("0", 1.00),
            ("a0", 1.00),
            ("b0", 1.00),
            ("c0", 1.00),
            ("d0", 1.00),
            ("e0", 1.00),
            ("a1", 1.00),
            ("b1", 1.00),
            ("c1", 1.00),
            ("d1", 1.00),
            ("e1", 1.00),
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
    fn test_opt8() {
        let edges = vec![
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
        ];

        let v_weight = HashMap::from([
            (0, 1.00),
            (1, 1.00),
            (2, 1.00),
            (3, 1.00),
            (4, 1.00),
            (5, 1.00),
            (6, 1.00),
            (7, 1.00),
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
    fn test_opt9() {
        let edges = vec![
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (8, 0),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (8, 1),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 2),
            (8, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
            (8, 8),
        ];

        let v_weight = HashMap::from([
            (0, 1.00),
            (1, 1.00),
            (2, 1.00),
            (3, 1.00),
            (4, 1.00),
            (5, 1.00),
            (6, 1.00),
            (7, 1.00),
            (8, 1.00),
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
    fn test15() {
        // BiSRGraph { u: 16, v_lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        // 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        // u_adj: [[0], [0, 1], [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 6],
        // [0, 6, 7], [0, 6, 8], [0, 6, 9], [0, 6, 10], [0, 11], [0, 11, 12],
        // [0, 11, 13], [0, 11, 14], [0, 11, 15]],
        // v_adj: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        // [1, 2, 3, 4, 5], [2], [3], [4], [5], [6, 7, 8, 9, 10],
        // [7], [8], [9], [10], [11, 12, 13, 14, 15], [12], [13], [14], [15]] }
        // opt = 16.0, ALG = 0.5993991440357337

        let edges = vec![
            ("X", "0"),
            ("A0", "0"),
            ("B0", "0"),
            ("C0", "0"),
            ("D0", "0"),
            ("E0", "0"),
            ("A1", "0"),
            ("B1", "0"),
            ("C1", "0"),
            ("D1", "0"),
            ("E1", "0"),
            ("A2", "0"),
            ("B2", "0"),
            ("C2", "0"),
            ("D2", "0"),
            ("E2", "0"),
            ("A0", "a0"),
            ("B0", "a0"),
            ("C0", "a0"),
            ("D0", "a0"),
            ("E0", "a0"),
            ("B0", "b0"),
            ("C0", "c0"),
            ("D0", "d0"),
            ("E0", "e0"),
            ("A1", "a1"),
            ("B1", "a1"),
            ("C1", "a1"),
            ("D1", "a1"),
            ("E1", "a1"),
            ("B1", "b1"),
            ("C1", "c1"),
            ("D1", "d1"),
            ("E1", "e1"),
            ("A2", "a2"),
            ("B2", "a2"),
            ("C2", "a2"),
            ("D2", "a2"),
            ("E2", "a2"),
            ("B2", "b2"),
            ("C2", "c2"),
            ("D2", "d2"),
            ("E2", "e2"),
        ];
        let v_weight = HashMap::from([
            ("0", 1.00),
            ("a0", 1.00),
            ("b0", 1.00),
            ("c0", 1.00),
            ("d0", 1.00),
            ("e0", 1.00),
            ("a1", 1.00),
            ("b1", 1.00),
            ("c1", 1.00),
            ("d1", 1.00),
            ("e1", 1.00),
            ("a2", 1.00),
            ("b2", 1.00),
            ("c2", 1.00),
            ("d2", 1.00),
            ("e2", 1.00),
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
    fn test16() {
        // opt = 7.0, ALG = 0.6059134644311833
        let edges = vec![
            ("X", "0"),
            ("A0", "0"),
            ("B0", "0"),
            ("C0", "0"),
            ("A1", "0"),
            ("B1", "0"),
            ("C1", "0"),
            ("A0", "a0"),
            ("B0", "a0"),
            ("C0", "a0"),
            ("B0", "b0"),
            ("C0", "c0"),
            ("A1", "a1"),
            ("B1", "a1"),
            ("C1", "a1"),
            ("B1", "b1"),
            ("C1", "c1"),
        ];
        let v_weight = HashMap::from([
            ("0", 1.00),
            ("a0", 1.00),
            ("b0", 1.00),
            ("c0", 1.00),
            ("a1", 1.00),
            ("b1", 1.00),
            ("c1", 1.00),
        ]);
        let mut opt = 0.;
        for (_, o) in &v_weight {
            opt += o;
        }
        let g = BiSRGraph::from_edge(edges, v_weight);

        println!("{:?}", g);

        println!("opt = {:?}, ALG = {:?}", opt, g.ALG() / opt);
    }
}
