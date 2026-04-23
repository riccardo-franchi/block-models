use rand_distr::{Distribution, Geometric};
use rayon::prelude::*;
use sbm_simulation::network_generation::create_degree_corrected_sbm;
use sbm_simulation::small_components::{
    compute_small_component_size_distribution, small_component_sizes,
};
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let avg_degree = |a: f64| a / (1.0 - a);

    let p_k = |a: f64, k: i32| (1.0 - a) * a.powi(k);
    let q_k = |a: f64, k: i32| (k + 1) as f64 * p_k(a, k + 1) / avg_degree(a);

    let a = [0.4, 0.8];
    let p1 = 0.999;
    let p2 = 1.0 - (1.0 - p1) * (avg_degree(a[0]) / avg_degree(a[1]));

    let max_size = 50;

    let psi = vec![vec![p1, 1.0 - p1], vec![1.0 - p2, p2]];

    let g0: Vec<Vec<f64>> = a
        .iter()
        .map(|&a_i| (0..max_size).map(|k| p_k(a_i, k as i32)).collect())
        .collect();

    let g1: Vec<Vec<f64>> = a
        .iter()
        .map(|&a_i| (0..max_size).map(|k| q_k(a_i, k as i32)).collect())
        .collect();

    let h0 = compute_small_component_size_distribution(&g0, &g1, &psi, max_size);

    let analytical: Vec<f64> = (1..=max_size)
        .map(|i| h0.iter().map(|h| h[i]).sum::<f64>() / h0.len() as f64)
        .collect();

    let num_trials = 100;
    let nodes_per_group = 1_000_000;

    let per_trial: Vec<(usize, Vec<usize>)> = (0..num_trials)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::rng();
            let geometric_distr: Vec<Geometric> = a
                .iter()
                .map(|&ai| Geometric::new(1.0 - ai).unwrap())
                .collect();

            let degree_sequence: Vec<Vec<usize>> = geometric_distr
                .iter()
                .map(|g| {
                    (0..nodes_per_group)
                        .map(|_| g.sample(&mut rng) as usize)
                        .collect()
                })
                .collect();

            let kappa: Vec<usize> = degree_sequence.iter().map(|seq| seq.iter().sum()).collect();

            let m12 = ((1.0 - p1) * kappa[0] as f64) as usize;
            let m = vec![
                vec![(kappa[0] - m12) / 2, m12],
                vec![m12, (kappa[1] - m12) / 2],
            ];

            let network = create_degree_corrected_sbm(&degree_sequence, &m);
            let sizes = small_component_sizes(&network);

            let total_small_nodes: usize = sizes.iter().sum();
            let mut component_counts = vec![0usize; max_size + 1];
            for &s in &sizes {
                if s <= max_size {
                    component_counts[s] += 1;
                }
            }
            (total_small_nodes, component_counts)
        })
        .collect();

    let total_small_nodes: usize = per_trial.iter().map(|(t, _)| *t).sum();
    let mut total_component_counts = vec![0usize; max_size + 1];
    for (_, counts) in &per_trial {
        for s in 1..=max_size {
            total_component_counts[s] += counts[s];
        }
    }

    std::fs::create_dir_all("output").expect("could not create output dir");
    let file = File::create("output/small_components.txt").expect("could not create output file");
    let mut writer = BufWriter::new(file);
    for (idx, s) in (1..=max_size).enumerate() {
        let sim = (s * total_component_counts[s]) as f64 / total_small_nodes as f64;

        writeln!(writer, "{s} {:.6} {:.6}", analytical[idx], sim).expect("write failed");
    }
}
