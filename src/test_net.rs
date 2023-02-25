#[cfg(test)]
mod tests_net {
    use crate::util::expected_success_distribution;

    #[test]
    fn test1() {
        println!("Hello, world!");
    }

    #[test]
    fn test_util_expected_success_distribution() {
        println!("{:?}", expected_success_distribution(2, 1.));
        println!("Hello, world!");
    }

}
