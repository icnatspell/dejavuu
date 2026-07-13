"""Unified benchmark entry point.

Use ``--dataset specbench|speedbench`` for text decoders and ``--dataset mmspec``
for the SmolVLM vision adapter. Remaining flags are forwarded unchanged to the
specialized runner, keeping the model-preparation seam out of dataset selection.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser("dejavuu.eval.bench", add_help=False)
    parser.add_argument("--dataset", choices=["specbench", "speedbench", "mmspec"], required=True)
    args, _ = parser.parse_known_args()
    if args.dataset == "mmspec":
        from dejavuu.eval.mmspec import main as run
    else:
        from dejavuu.eval.specbench import main as run
    run()


if __name__ == "__main__":
    main()
