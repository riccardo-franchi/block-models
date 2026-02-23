use rand::Rng;
use rand_distr::{Distribution, Poisson};
use rayon::prelude::*;
use statrs::distribution::{Binomial, Discrete};
use statrs::statistics::Statistics;
use std::fs::File;
use std::io::{BufWriter, Write};

// Adjacency list representation
type Network = Vec<Vec<usize>>;

fn create_stochastic_block_model(n: &[usize], m: &[Vec<usize>]) -> Network {
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

fn sample_edges(n: &[usize], c: &[Vec<f64>]) -> Vec<Vec<usize>> {
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

fn get_edges(network: &Network) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for (i, neighbors) in network.iter().enumerate() {
        for &neighbor in neighbors {
            if i < neighbor {
                edges.push((i, neighbor));
            }
        }
    }

    edges
}

fn union_find_root(i: usize, pointers: &mut [i32]) -> usize {
    let mut r = i;
    let mut s = i;

    while pointers[r] >= 0 {
        pointers[s] = pointers[r];
        s = r;
        r = pointers[r] as usize;
    }

    r
}

fn sweep_percolation(network: &Network) -> Vec<f64> {
    let mut edges_order = get_edges(network);

    // shuffle edges using Fisher-Yates algorithm
    let mut rng = rand::rng();
    for i in (1..edges_order.len()).rev() {
        let j = rng.random_range(0..=i);
        edges_order.swap(i, j);
    }

    // perform percolation
    let mut pointers: Vec<i32> = vec![-1; network.len()];
    let mut s_r = Vec::with_capacity(edges_order.len());

    let mut biggest_cluster_size = 1;

    for nodes in edges_order {
        let root1 = union_find_root(nodes.0, &mut pointers);
        let root2 = union_find_root(nodes.1, &mut pointers);
        if root1 != root2 {
            if pointers[root1] > pointers[root2] {
                pointers[root2] += pointers[root1];
                pointers[root1] = root2 as i32;
            } else {
                pointers[root1] += pointers[root2];
                pointers[root2] = root1 as i32;
            }
            if -pointers[root1] > biggest_cluster_size {
                biggest_cluster_size = -pointers[root1];
            }
        }

        s_r.push(biggest_cluster_size as f64 / network.len() as f64);
    }

    s_r
}

fn calc_s_phi(s_r: &[f64], num_points: usize) -> Vec<f64> {
    let n = s_r.len();
    let mut s_phi = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let phi = i as f64 / (num_points - 1) as f64;

        // Sum from r=0 to n: Binomial(n, phi).pmf(r) * S_r
        // S_0 (no edges) = 0, S_r for r>0 is s_r[r-1]
        let sum = if phi == 0.0 {
            0.0
        } else if phi == 1.0 {
            s_r[n - 1]
        } else {
            let binom = Binomial::new(phi, n as u64).unwrap();
            (1..=n).map(|r| binom.pmf(r as u64) * s_r[r - 1]).sum()
        };

        s_phi.push(sum);
    }

    s_phi
}

fn main() {
    let b = 0.001;

    let c = vec![vec![4. / 3., b, b], vec![b, 5.0, b], vec![b, b, 20.0]];
    let n = vec![100_000; 3];

    let num_points = 80;
    let num_trials = 20;

    // Collect s_phi across trials
    let all_s_phi: Vec<Vec<f64>> = (0..num_trials)
        .into_par_iter()
        .map(|_| {
            let m = sample_edges(&n, &c);
            let network = create_stochastic_block_model(&n, &m);
            let s_r = sweep_percolation(&network);
            calc_s_phi(&s_r, num_points)
        })
        .collect();

    // For each phi index compute mean and std dev of the mean across trials
    let file = File::create("s_phi.txt").expect("could not create s_phi.txt");
    let mut writer = BufWriter::new(file);
    for i in 0..num_points {
        let phi = i as f64 / (num_points - 1) as f64;
        let values: Vec<f64> = all_s_phi.iter().map(|s| s[i]).collect();
        let mean = values.iter().mean();
        let std_of_mean = values.iter().std_dev() / (num_trials as f64).sqrt();
        writeln!(writer, "{phi:.6} {mean:.6} {std_of_mean:.6}").expect("write failed");
    }
}
