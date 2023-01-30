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

    test15();
}


fn test15() {
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
    let g = bisrgraph::BiSRGraph::from_edge(edges, v_weight);

    println!("{:?}", g);

    println!("opt = {:?}, ALG = {:?}", opt, g.ALG() / opt);
}