#!/bin/bash
set -e

cargo run --release && .venv/bin/python3 plot.py
