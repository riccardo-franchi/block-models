use crate::Network;
use rand::Rng;
use rand_distr::{Distribution, Poisson};

pub fn create_stochastic_block_model(n: &[usize], m: &[Vec<usize>]) -> Network {
    let num_groups = n.len();
    let num_nodes = n.iter().sum();
    let mut network: Network = vec![Vec::new(); num_nodes];
    let mut rng = rand::rng();

    for r in 0..num_groups {
        for s in r..num_groups {
            let mut edges_added = 0;
            while edges_added < m[r][s] {
                let a = rng.random_range(..n[r]) + n[..r].iter().sum::<usize>();
                let b = rng.random_range(..n[s]) + n[..s].iter().sum::<usize>();

                if a != b && !network[a].contains(&b) {
                    network[a].push(b);
                    network[b].push(a);
                    edges_added += 1;
                }
            }
        }
    }

    network
}

pub fn sample_edges(n: &[usize], c: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let mut m = vec![vec![0; c.len()]; c.len()];
    let mut rng = rand::rng();

    for r in 0..c.len() {
        let mean_rr = c[r][r] * n[r] as f64 / 2.0;
        let poisson_rr = Poisson::new(mean_rr).unwrap();

        m[r][r] = poisson_rr.sample(&mut rng) as usize;

        for s in r + 1..c.len() {
            let mean_rs = c[r][s] * n[r] as f64;
            let poisson_rs = Poisson::new(mean_rs).unwrap();
            m[r][s] = poisson_rs.sample(&mut rng) as usize;
            m[s][r] = m[r][s];
        }
    }

    m
}
