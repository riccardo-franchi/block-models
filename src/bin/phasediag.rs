use rand_distr::{Distribution, Geometric};
use rayon::prelude::*;
use sbm_simulation::network_generation::create_microcanonical_sbm;
use sbm_simulation::percolation::calc_largest_component_size;
use std::fs::File;
use std::io::{BufWriter, Write};

const GRID: usize = 200;
const NUM_TRIALS: usize = 100;
const NUM_NODES: usize = 100_000;

const IMPOSSIBLE: f64 = -1.0;

fn giant_fraction(c1: f64, c2: f64) -> f64 {
    let nodes_per_group = NUM_NODES / 2;
    let t = ((NUM_NODES as f64) / 8.0).round() as usize;

    // Geometric with success probability 1 / (1 + c) has mean degree c.
    let geometric = [
        Geometric::new(1.0 / (1.0 + c1)).unwrap(),
        Geometric::new(1.0 / (1.0 + c2)).unwrap(),
    ];

    let mut total = 0.0;
    let mut valid = 0usize;

    for _ in 0..NUM_TRIALS {
        let degree_sequence: Vec<Vec<usize>> = (0..2)
            .map(|g| {
                (0..nodes_per_group)
                    .map(|_| geometric[g].sample(&mut rand::rng()) as usize)
                    .collect()
            })
            .collect();

        let kappa: Vec<usize> = degree_sequence.iter().map(|seq| seq.iter().sum()).collect();

        if kappa[0] < t || kappa[1] < t {
            continue;
        }

        let m = vec![vec![(kappa[0] - t) / 2, t], vec![t, (kappa[1] - t) / 2]];

        let network = create_microcanonical_sbm(&degree_sequence, &m);
        total += calc_largest_component_size(&network) as f64 / network.len() as f64;
        valid += 1;
    }

    if valid == 0 {
        IMPOSSIBLE
    } else {
        total / valid as f64
    }
}

fn main() {
    // Row index -> group-2 mean degree c2, column index -> group-1 mean degree c1
    let mut flat: Vec<f64> = vec![0.0; GRID * GRID];
    flat.par_iter_mut().enumerate().for_each(|(idx, out)| {
        let i = idx / GRID;
        let j = idx % GRID;
        let c2 = i as f64 / (GRID - 1) as f64;
        let c1 = j as f64 / (GRID - 1) as f64;
        *out = giant_fraction(c1, c2);
    });

    std::fs::create_dir_all("output").expect("could not create output dir");
    let file = File::create("output/phasediag.txt").expect("could not create phasediag.txt");
    let mut writer = BufWriter::new(file);
    for row in flat.chunks(GRID) {
        let line: Vec<String> = row.iter().map(|v| format!("{v:.6}")).collect();
        writeln!(writer, "{}", line.join(" ")).expect("write failed");
    }
}
