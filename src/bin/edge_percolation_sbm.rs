use rayon::prelude::*;
use sbm_simulation::network_generation::{create_stochastic_block_model, sample_poisson_edges};
use sbm_simulation::percolation::{calc_s_phi, sweep_edge_percolation};
use statrs::statistics::Statistics;
use std::fs::File;
use std::io::{BufWriter, Write};

fn run(b: f64, num_nodes: f64, num_points: usize, num_trials: usize) -> Vec<(f64, f64, f64)> {
    let c = [vec![2.0, b], vec![b, 20.0]];
    let n = [(num_nodes * 0.5) as usize, (num_nodes * 0.5) as usize];

    let all_s_phi: Vec<Vec<f64>> = (0..num_trials)
        .into_par_iter()
        .map(|_| {
            let m = sample_poisson_edges(&n, &c);
            let network = create_stochastic_block_model(&n, &m);
            let s_r = sweep_edge_percolation(&network);
            calc_s_phi(&s_r, num_points)
        })
        .collect();

    (0..num_points)
        .map(|i| {
            let phi = i as f64 / (num_points - 1) as f64;
            let values: Vec<f64> = all_s_phi.iter().map(|s| s[i]).collect();
            let mean = values.iter().mean();
            let std_of_mean = values.iter().std_dev() / (num_trials as f64).sqrt();
            (phi, mean, std_of_mean)
        })
        .collect()
}

fn main() {
    let num_nodes = 500_000.0;
    let num_points = 40;
    let num_trials = 20;

    let b_values = [0.1, 0.001];

    std::fs::create_dir_all("output").expect("could not create output dir");
    for b in b_values {
        let results = run(b, num_nodes, num_points, num_trials);
        let path = format!("output/edge_percolation_sbm_b{b}.txt");
        let file = File::create(&path).expect("could not create output file");
        let mut writer = BufWriter::new(file);
        for (phi, mean, std_of_mean) in results {
            writeln!(writer, "{phi:.6} {mean:.6} {std_of_mean:.6}").expect("write failed");
        }
    }
}
