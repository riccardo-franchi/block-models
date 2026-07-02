#!/bin/bash
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <experiment>" >&2
    echo "  experiment: one of" >&2
    echo "    phasediag" >&2
    echo "    edge_percolation_sbm" >&2
    echo "    edge_percolation_dc_sbm" >&2
    echo "    targeted_percolation_dc_sbm" >&2
    echo "    small_components" >&2
    exit 1
fi

name="$1"
cargo run --release --bin "$name" && .venv/bin/python3 "plot_${name}.py"
