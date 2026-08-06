"""Pre-download + cache real Perturb-seq datasets once, so cluster compute jobs
read from disk instead of hitting the network (compute nodes are often offline).

Downloads via the bio-perturbations dataset adapters (pertpy under the hood) into
--cache-dir. Later runs read the same cache by passing the SAME --cache-dir to
scldm_bio_eval.py / scldm_ablation.py.
"""

from __future__ import annotations

import argparse

from bio_perturbations.datasets import load_dataset, list_datasets


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["replogle_2022_k562"],
                    help=f"ids to fetch. available: {list_datasets()}")
    ap.add_argument("--cache-dir", required=True,
                    help="download/cache directory (reuse this in the run jobs)")
    args = ap.parse_args()

    for name in args.datasets:
        print(f"[download] {name} -> {args.cache_dir}", flush=True)
        adata = load_dataset(name, cache_dir=args.cache_dir, require_raw_counts=True)
        n_perts = adata.obs["condition"].astype(str).nunique()
        print(f"[done] {name}: {adata.n_obs} cells x {adata.n_vars} genes, "
              f"{n_perts} conditions", flush=True)
    print("[all datasets cached]")


if __name__ == "__main__":
    main()
