use rand_distr::{Distribution, Geometric, Zeta};
use rayon::prelude::*;
use sbm_simulation::network_generation::create_degree_corrected_sbm;
use sbm_simulation::percolation::{calc_s_phi, sweep_edge_percolation};
use statrs::statistics::Statistics;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let a = 0.5;
    let alpha = 2.5;

    let p1 = 0.999;

    let geometric = Geometric::new(1.0 - a).unwrap();
    let power = Zeta::new(alpha).unwrap();

    let num_nodes = 500_000.0;
    let n = [(num_nodes * 0.9) as usize, (num_nodes * 0.1) as usize];

    let num_points = 80;
    let num_trials = 10;

    let all_s_phi: Vec<Vec<f64>> = (0..num_trials)
        .into_par_iter()
        .map(|_| {
            let mut degree_sequence = (0..n.len())
                .map(|i| Vec::with_capacity(n[i]))
                .collect::<Vec<_>>();

            for _ in 0..n[0] {
                degree_sequence[0].push(geometric.sample(&mut rand::rng()) as usize);
            }
            for _ in 0..n[1] {
                degree_sequence[1].push(power.sample(&mut rand::rng()) as usize);
            }

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
            let s_r = sweep_edge_percolation(&network);
            calc_s_phi(&s_r, num_points)
        })
        .collect();

    std::fs::create_dir_all("output").expect("could not create output dir");
    let file =
        File::create("output/edge_percolation_dc_sbm.txt").expect("could not create output file");
    let mut writer = BufWriter::new(file);
    for i in 0..num_points {
        let phi = i as f64 / (num_points - 1) as f64;
        let values: Vec<f64> = all_s_phi.iter().map(|s| s[i]).collect();
        let mean = values.iter().mean();
        let std_of_mean = values.iter().std_dev() / (num_trials as f64).sqrt();
        writeln!(writer, "{phi:.6} {mean:.6} {std_of_mean:.6}").expect("write failed");
    }
}
