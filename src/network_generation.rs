use crate::Network;
use rand::RngExt;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Poisson};

pub fn create_stochastic_block_model(n: &[usize], m: &[Vec<usize>]) -> Network {
    let num_groups = n.len();
    let num_nodes = n.iter().sum();
    let mut network: Network = vec![Vec::new(); num_nodes];
    let mut rng = rand::rng();

    let offsets: Vec<usize> = n
        .iter()
        .scan(0, |acc, &size| {
            let offset = *acc;
            *acc += size;
            Some(offset)
        })
        .collect();

    for r in 0..num_groups {
        for s in r..num_groups {
            let mut edges_added = 0;
            while edges_added < m[r][s] {
                let a = rng.random_range(..n[r]) + offsets[r];
                let b = rng.random_range(..n[s]) + offsets[s];

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

pub fn sample_poisson_edges(n: &[usize], c: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let mut m = vec![vec![0; c.len()]; c.len()];
    let mut rng = rand::rng();

    for r in 0..c.len() {
        m[r][r] = Poisson::new(c[r][r] * n[r] as f64 / 2.0)
            .unwrap()
            .sample(&mut rng) as usize;

        for s in r + 1..c.len() {
            let count = Poisson::new(c[r][s] * n[r] as f64)
                .unwrap()
                .sample(&mut rng) as usize;
            m[r][s] = count;
            m[s][r] = count;
        }
    }

    m
}

pub fn create_microcanonical_sbm(degree_sequence: &[Vec<usize>], m: &[Vec<usize>]) -> Network {
    let num_nodes = degree_sequence.iter().map(|seq| seq.len()).sum();
    let num_groups = degree_sequence.len();
    let mut network: Network = vec![Vec::new(); num_nodes];

    let mut rng = rand::rng();

    let group_offsets: Vec<usize> = degree_sequence
        .iter()
        .scan(0, |acc, seq| {
            let offset = *acc;
            *acc += seq.len();
            Some(offset)
        })
        .collect();

    let stubs: Vec<Vec<usize>> = degree_sequence
        .iter()
        .enumerate()
        .map(|(r, degrees)| {
            let mut group_stubs: Vec<usize> = degrees
                .iter()
                .enumerate()
                .flat_map(|(i, &deg)| std::iter::repeat_n(group_offsets[r] + i, deg))
                .collect();
            group_stubs.shuffle(&mut rng);
            group_stubs
        })
        .collect();

    let mut stub_pointers: Vec<usize> = vec![0; num_groups];

    for r in 0..num_groups {
        for i in 0..m[r][r] {
            let a = stubs[r][stub_pointers[r] + 2 * i];
            let b = stubs[r][stub_pointers[r] + 2 * i + 1];
            network[a].push(b);
            network[b].push(a);
        }
        stub_pointers[r] += m[r][r] * 2;

        for s in r + 1..num_groups {
            for i in 0..m[r][s] {
                let a = stubs[r][stub_pointers[r] + i];
                let b = stubs[s][stub_pointers[s] + i];
                network[a].push(b);
                network[b].push(a);
            }
            stub_pointers[r] += m[r][s];
            stub_pointers[s] += m[r][s];
        }
    }

    network
}
