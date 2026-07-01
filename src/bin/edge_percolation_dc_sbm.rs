use rand_distr::{Distribution, Geometric, Zeta};
use rayon::prelude::*;
use sbm_simulation::network_generation::create_microcanonical_sbm;
use sbm_simulation::percolation::{calc_s_phi, sweep_edge_percolation};
use statrs::statistics::Statistics;
use std::fs::File;
use std::io::{BufWriter, Write};

const NUM_POINTS: usize = 40;
const NUM_TRIALS: usize = 20;

fn write_output(all_s_phi: &[Vec<f64>], path: &str) {
    std::fs::create_dir_all("output").expect("could not create output dir");
    let file = File::create(path).expect("could not create output file");
    let mut writer = BufWriter::new(file);
    for i in 0..NUM_POINTS {
        let phi = i as f64 / (NUM_POINTS - 1) as f64;
        let values: Vec<f64> = all_s_phi.iter().map(|s| s[i]).collect();
        let mean = values.iter().mean();
        let std_of_mean = values.iter().std_dev() / (NUM_TRIALS as f64).sqrt();
        writeln!(writer, "{phi:.6} {mean:.6} {std_of_mean:.6}").expect("write failed");
    }
}

fn run_geometric(a: [f64; 2], num_nodes: f64, p1: f64) -> Vec<Vec<f64>> {
    let n = [(num_nodes * 0.5) as usize, (num_nodes * 0.5) as usize];

    (0..NUM_TRIALS)
        .into_par_iter()
        .map(|_| {
            let geometric: Vec<Geometric> = a
                .iter()
                .map(|&ai| Geometric::new(1.0 - ai).unwrap())
                .collect();

            let degree_sequence: Vec<Vec<usize>> = (0..2)
                .map(|i| {
                    (0..n[i])
                        .map(|_| geometric[i].sample(&mut rand::rng()) as usize)
                        .collect()
                })
                .collect();

            let kappa = degree_sequence
                .iter()
                .map(|seq| seq.iter().sum::<usize>())
                .collect::<Vec<usize>>();

            let m12 = ((1.0 - p1) * kappa[0] as f64) as usize;
            let m = vec![
                vec![(kappa[0] - m12) / 2, m12],
                vec![m12, (kappa[1] - m12) / 2],
            ];

            let network = create_microcanonical_sbm(&degree_sequence, &m);
            let s_r = sweep_edge_percolation(&network);
            calc_s_phi(&s_r, NUM_POINTS)
        })
        .collect()
}

fn run_geometric_power(a: f64, alpha: f64, num_nodes: f64, p1: f64) -> Vec<Vec<f64>> {
    let n = [(num_nodes * 0.9) as usize, (num_nodes * 0.1) as usize];

    (0..NUM_TRIALS)
        .into_par_iter()
        .map(|_| {
            let geometric = Geometric::new(1.0 - a).unwrap();
            let power = Zeta::new(alpha).unwrap();

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

            let network = create_microcanonical_sbm(&degree_sequence, &m);
            let s_r = sweep_edge_percolation(&network);
            calc_s_phi(&s_r, NUM_POINTS)
        })
        .collect()
}

fn main() {
    let num_nodes = 500_000.0;
    let p1 = 0.999;

    let geometric = run_geometric([0.4, 0.8], num_nodes, p1);
    write_output(&geometric, "output/edge_percolation_dc_sbm_geometric.txt");

    let mixed = run_geometric_power(0.5, 2.5, num_nodes, p1);
    write_output(&mixed, "output/edge_percolation_dc_sbm_mixed.txt");
}
