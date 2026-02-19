#!/usr/bin/env python3
"""Configuration loader for CVAE PCB Warpage project.

Adapted from references/load_config.py — parses the key-value config.txt format.
"""

import os


def load_config(config_path):
    """Load configuration from config.txt."""
    config = {}

    print(f"Loading configuration from {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and section headers
            if not line or line.startswith('%'):
                continue

            # Strip inline comments
            if '#' in line:
                line = line.split('#')[0].strip()

            if not line:
                continue

            # Support both tab and space separation
            if '\t' in line:
                parts = line.split('\t')
            else:
                parts = line.split()

            if len(parts) >= 2:
                key = parts[0].strip().lower()
                value = ' '.join(parts[1:]).strip()
                config[key] = _parse_value(value)

    print(f"Configuration loaded: {len(config)} parameters")
    return config


def _parse_value(value_str):
    """Parse a string value to an appropriate Python type."""
    value_str = value_str.strip()

    # Comma-separated list (e.g. gpu_ids "0, 1")
    if ',' in value_str:
        parts = [p.strip() for p in value_str.split(',')]
        try:
            return [int(p) if '.' not in p else float(p) for p in parts]
        except ValueError:
            return [p.lower() for p in parts]

    # Boolean
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'

    # Numeric
    try:
        if '.' in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        return value_str.lower()


def display_config(config):
    """Pretty-print all configuration parameters."""
    print("\n" + "=" * 60)
    print("CONFIGURATION PARAMETERS")
    print("=" * 60)
    for key, value in sorted(config.items()):
        print(f"  {key:<25}: {value}  ({type(value).__name__})")
    print("=" * 60)
    print(f"Total: {len(config)} parameters")
    print("=" * 60 + "\n")
