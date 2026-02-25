use crate::Network;
use rand::Rng;
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

pub fn sweep_percolation(network: &Network) -> Vec<f64> {
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

pub fn calc_s_phi(s_r: &[f64], num_points: usize) -> Vec<f64> {
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
