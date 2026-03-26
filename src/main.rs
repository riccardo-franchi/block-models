use rand_distr::{Distribution, Poisson};
use rayon::prelude::*;
use statrs::statistics::Statistics;
use std::fs::File;
use std::io::{BufWriter, Write};

mod network_generation;
mod percolation;

use network_generation::create_degree_corrected_sbm;
use percolation::{calc_s_phi, sweep_edge_percolation};

// Adjacency list representation
pub type Network = Vec<Vec<usize>>;

fn main() {
    let c = (2.1, 20.1);

    let poisson1 = Poisson::new(c.0).unwrap();
    let poisson2 = Poisson::new(c.1).unwrap();

    let nodes_per_group = 100_000;

    let num_points = 80;
    let num_trials = 10;

    // Collect s_phi across trials
    let all_s_phi: Vec<Vec<f64>> = (0..num_trials)
        .into_par_iter()
        .map(|_| {
            let mut degree_sequence = (0..2)
                .map(|_| Vec::with_capacity(nodes_per_group))
                .collect::<Vec<_>>();

            for _ in 0..nodes_per_group {
                degree_sequence[0].push(poisson1.sample(&mut rand::rng()) as usize);
                degree_sequence[1].push(poisson2.sample(&mut rand::rng()) as usize);
            }

            // sum of degrees of each group
            let kappa = degree_sequence
                .iter()
                .map(|seq| seq.iter().sum::<usize>())
                .collect::<Vec<usize>>();

            let psi_12 = 0.1 / c.0;
            let m01 = (psi_12 * kappa[0] as f64) as usize;

            let m = vec![
                vec![(kappa[0] - m01) / 2, m01],
                vec![m01, (kappa[1] - m01) / 2],
            ];

            let network = create_degree_corrected_sbm(&degree_sequence, &m);
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
