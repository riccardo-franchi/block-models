use crate::Network;
use rand::seq::SliceRandom;
use statrs::distribution::{Binomial, Discrete};

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

fn get_edges(network: &Network) -> Vec<(usize, usize)> {
    network
        .iter()
        .enumerate()
        .flat_map(|(i, neighbors)| {
            neighbors
                .iter()
                .filter(move |&&j| i < j)
                .map(move |&j| (i, j))
        })
        .collect()
}

pub fn sweep_edge_percolation(network: &Network) -> Vec<f64> {
    let mut edges_order = get_edges(network);

    let mut rng = rand::rng();
    edges_order.shuffle(&mut rng);

    // perform percolation
    let mut pointers: Vec<i32> = vec![-1; network.len()];
    let mut s_r = Vec::with_capacity(edges_order.len());

    let mut biggest_cluster_size = 1;

    for (node_a, node_b) in edges_order {
        let root1 = union_find_root(node_a, &mut pointers);
        let root2 = union_find_root(node_b, &mut pointers);
        if root1 != root2 {
            let root = if pointers[root1] > pointers[root2] {
                pointers[root2] += pointers[root1];
                pointers[root1] = root2 as i32;
                root2
            } else {
                pointers[root1] += pointers[root2];
                pointers[root2] = root1 as i32;
                root1
            };
            if -pointers[root] > biggest_cluster_size {
                biggest_cluster_size = -pointers[root];
            }
        }

        s_r.push(biggest_cluster_size as f64 / network.len() as f64);
    }

    s_r
}

pub fn calc_s_phi(s_r: &[f64], num_points: usize) -> Vec<f64> {
    let n = s_r.len();

    // Sum from r=0 to n: Binomial(n, phi).pmf(r) * S_r
    // S_0 (no edges) = 0, S_r for r>0 is s_r[r-1]
    (0..num_points)
        .map(|i| {
            let phi = i as f64 / (num_points - 1) as f64;
            if phi == 0.0 {
                0.0
            } else if phi == 1.0 {
                s_r[n - 1]
            } else {
                let binom = Binomial::new(phi, n as u64).unwrap();
                (1..=n).map(|r| binom.pmf(r as u64) * s_r[r - 1]).sum()
            }
        })
        .collect()
}
