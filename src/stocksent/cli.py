"""Command line: stocksent {ingest,features,train,report,all,serve} [--config path]."""
from __future__ import annotations

import os

# keep BLAS thread pools small before numpy is imported; the full pipeline then fits in a few GB of RAM
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stocksent", description=__doc__)
    parser.add_argument("stage", choices=["ingest", "features", "train", "report", "all", "serve"])
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.ERROR)

    if args.stage == "serve":
        import uvicorn

        uvicorn.run("stocksent.api.app:app", host="0.0.0.0", port=args.port)
        return

    from . import pipeline, report
    from .config import load_config

    cfg = load_config(args.config)
    stages = ["ingest", "features", "train", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        t0 = time.time()
        {"ingest": pipeline.ingest, "features": pipeline.build_features, "train": pipeline.train,
         "report": report.build_report}[stage](cfg)
        logging.info("stage %s finished in %.0fs", stage, time.time() - t0)


if __name__ == "__main__":
    main()
