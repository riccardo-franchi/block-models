use rayon::prelude::*;
use statrs::statistics::Statistics;
use std::fs::File;
use std::io::{BufWriter, Write};

mod network_generation;
mod percolation;

use network_generation::{create_stochastic_block_model, sample_poisson_edges};
use percolation::{calc_s_phi, sweep_edge_percolation};

// Adjacency list representation
pub type Network = Vec<Vec<usize>>;

fn main() {
    let b = 0.1;

    let c = vec![vec![2.0, b], vec![b, 20.0]];
    let n = vec![100_000; 2];

    let num_points = 80;
    let num_trials = 10;

    // Collect s_phi across trials
    let all_s_phi: Vec<Vec<f64>> = (0..num_trials)
        .into_par_iter()
        .map(|_| {
            let m = sample_poisson_edges(&n, &c);
            let network = create_stochastic_block_model(&n, &m);
            let s_r = sweep_edge_percolation(&network);
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
