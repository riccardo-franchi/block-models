use crate::Network;
use std::collections::VecDeque;

pub fn small_component_sizes(network: &Network) -> Vec<usize> {
    let mut visited = vec![false; network.len()];
    let mut sizes = Vec::new();

    for node in 0..network.len() {
        if visited[node] {
            continue;
        }

        let mut queue = VecDeque::from([node]);
        visited[node] = true;
        let mut component_size = 0;

        while let Some(current) = queue.pop_front() {
            component_size += 1;
            for &neighbor in &network[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        sizes.push(component_size);
    }

    sizes
}

fn polynomial_multiply(a: &[f64], b: &[f64], max_len: usize) -> Vec<f64> {
    let mut result = vec![0.0; max_len];
    for (i, &a_i) in a.iter().enumerate().take(max_len) {
        if a_i == 0.0 {
            continue;
        }
        for (j, &b_j) in b.iter().enumerate().take(max_len - i) {
            result[i + j] += a_i * b_j;
        }
    }
    result
}

fn horner_compose(g: &[f64], y: &[f64], max_len: usize) -> Vec<f64> {
    let mut it = g.iter().rev();
    let mut acc = vec![0.0; max_len];
    acc[0] = it.next().copied().unwrap();
    it.fold(acc, |mut acc, &gi| {
        acc = polynomial_multiply(&acc, y, max_len);
        acc[0] += gi;
        acc
    })
}

fn shift_right(p: &[f64], max_len: usize) -> Vec<f64> {
    let mut result = vec![0.0; max_len + 1];
    result[1..=max_len].copy_from_slice(&p[..max_len]);
    result
}

fn psi_times_h(psi: &[Vec<f64>], h: &[Vec<f64>], len: usize) -> Vec<Vec<f64>> {
    psi.iter()
        .map(|psi_r| {
            psi_r
                .iter()
                .zip(h)
                .fold(vec![0.0; len], |mut y_r, (&psi_rs, h_s)| {
                    y_r.iter_mut()
                        .zip(&h_s[..])
                        .for_each(|(out, &rho)| *out += psi_rs * rho);
                    y_r
                })
        })
        .collect()
}

pub fn compute_small_component_size_distribution(
    g0: &[Vec<f64>],
    g1: &[Vec<f64>],
    psi: &[Vec<f64>],
    max_size: usize,
) -> Vec<Vec<f64>> {
    assert!(g0.iter().all(|g| g.len() >= max_size));
    assert!(g1.iter().all(|g| g.len() >= max_size - 1));

    let mut h1: Vec<Vec<f64>> = vec![vec![0.0]; g0.len()];
    for i in 1..max_size {
        let y = psi_times_h(psi, &h1, i);
        h1 = g1
            .iter()
            .zip(&y)
            .map(|(g1_r, y_r)| shift_right(&horner_compose(&g1_r[..i], y_r, i), i))
            .collect();
    }

    let y = psi_times_h(psi, &h1, max_size);
    g0.iter()
        .zip(&y)
        .map(|(g0_r, y_r)| shift_right(&horner_compose(&g0_r[..max_size], y_r, max_size), max_size))
        .collect()
}
