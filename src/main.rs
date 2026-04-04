#![allow(dead_code)]

use rand_distr::{Distribution, Geometric};
use rayon::prelude::*;
use statrs::statistics::Statistics;
use std::fs::File;
use std::io::{BufWriter, Write};

mod network_generation;
mod percolation;

use network_generation::create_degree_corrected_sbm;
use percolation::targeted_node_percolation;

// Adjacency list representation
pub type Network = Vec<Vec<usize>>;

fn main() {
    let a = (0.5, 0.95);
    let p1 = 0.999;

    let geometric1 = Geometric::new(1.0 - a.0).unwrap();
    let geometric2 = Geometric::new(1.0 - a.1).unwrap();

    let nodes_per_group = 100_000;

    let num_trials = 5;

    let all_results: Vec<Vec<(f64, f64)>> = (0..num_trials)
        .into_par_iter()
        .map(|_| {
            let mut degree_sequence = (0..2)
                .map(|_| Vec::with_capacity(nodes_per_group))
                .collect::<Vec<_>>();

            for _ in 0..nodes_per_group {
                degree_sequence[0].push(geometric1.sample(&mut rand::rng()) as usize);
                degree_sequence[1].push(geometric2.sample(&mut rand::rng()) as usize);
            }

            // sum of degrees of each group
            let kappa = degree_sequence
                .iter()
                .map(|seq| seq.iter().sum::<usize>())
                .collect::<Vec<usize>>();

            let m12 = ((1.0 - p1) * kappa[0] as f64) as usize;

            let m = vec![
                vec![(kappa[0] - m12) / 2, m12],
                vec![m12, (kappa[1] - m12) / 2],
            ];

            let network = create_degree_corrected_sbm(&degree_sequence, &m);
            targeted_node_percolation(&network)
        })
        .collect();

    // For each k compute mean and std dev of the mean across trials
    let num_points = all_results.iter().map(|v| v.len()).min().unwrap();
    let file = File::create("s_k.txt").expect("could not create s_k.txt");
    let mut writer = BufWriter::new(file);
    for k in 0..num_points {
        let occupied_values: Vec<f64> = all_results.iter().map(|r| r[k].0).collect();
        let s_values: Vec<f64> = all_results.iter().map(|r| r[k].1).collect();
        let mean_occupied = occupied_values.iter().mean();
        let mean_s = s_values.iter().mean();
        writeln!(writer, "{k} {mean_occupied:.6} {mean_s:.6}").expect("write failed");
    }
}
