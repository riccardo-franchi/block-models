use rand::Rng;
use rand_distr::{Distribution, Poisson};
use std::collections::VecDeque;

// Adjacency list representation
type Network = Vec<Vec<usize>>;

fn create_stochastic_block_model(n: &Vec<usize>, m: &Vec<Vec<usize>>) -> Network {
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

fn calc_largest_component_size(network: &Network) -> usize {
    let mut visited = vec![false; network.len()];
    let mut largest_size = 0;

    for node in 0..network.len() {
        if visited[node] {
            continue;
        }

        let mut queue = VecDeque::from([node]);
        visited[node] = true;
        let mut component_size = 0;

        // perform BFS to find all nodes in this component
        while let Some(current) = queue.pop_front() {
            component_size += 1;

            for &neighbor in &network[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        if component_size > largest_size {
            largest_size = component_size;
        }
    }

    largest_size
}

fn main() {
    let b = 0.01;

    let c = vec![vec![2.0, b], vec![b, 20.0]];

    // Construct n and m
    let n = vec![1_000_000; 2];
    let mut m = vec![vec![0; c.len()]; c.len()];
    let mut rng = rand::rng();

    let num_realizations = 10;
    let mut total_largest_component_size = 0;

    for _ in 0..num_realizations {
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

        let network = create_stochastic_block_model(&n, &m);
        total_largest_component_size += calc_largest_component_size(&network);
    }

    println!(
        "Avg largest component over {} realizations: {}",
        num_realizations,
        (total_largest_component_size as f64 / num_realizations as f64)
            / n.iter().sum::<usize>() as f64
    );
}
