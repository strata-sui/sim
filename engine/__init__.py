"""Strata sim — engine package.

Phase scope:
- S0: data loader, resampler, data-sanity gate (this commit)
- S2: rigorous price engine (block bootstrap + Kou jump overlay)
- S3: stochastic SVI engine (OU + no-arb gate)
"""
