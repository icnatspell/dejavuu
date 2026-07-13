"""SmolVLM2 benchmark: baseline vs methods on text (Spec-Bench) or vision (MMSpec).

    uv run --extra vlm python -m dejavuu.eval.mmspec --dataset mmspec \
        --methods baseline,pld,token_recycling --per-category 3 --max-new 128

One model (SmolVLM2) for both runs: `--dataset specbench` feeds text-only prompts,
`--dataset mmspec` feeds image + question. Spec-decode uses the genai-built decoder
(vlm.py auto-loads it; the stock decoder is locked to seq_len=1). That decoder's
int4 GQA kernel isn't length-invariant, so spec-decode is near- (not strictly)
lossless -- the table reports a token-match % vs baseline (see model contract).
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from dejavuu.decoders.vlm import GENAI_DECODER, VLM, VLM_TREE_DECODER, download
from dejavuu.eval.harness import (
    benchmark_metadata,
    create_run_dir,
    load_datastore,
    make_drafter,
    render_table,
    write_car_profile,
    write_response_jsonl,
    write_run_manifest,
)
from dejavuu.eval.runner import RunCase, run_cases
from dejavuu.eval.specbench import load_specbench

RAW = "https://raw.githubusercontent.com/killthefullmoon/MMSpec/main/dataset/MMSpec/testmini"
CACHE = Path.home() / ".cache" / "dejavuu" / "mmspec"

# Report by MMSpec's `topic` field (the dataset's own task taxonomy, 6 x 10) rather
# than `category` (which mixes task type with subject domain); rename to short labels.
MMSPEC_TOPIC = {
    "chart understanding": "chart vqa",
    "complex reasoning pro": "complex reasoning",
    "multi-turn conversation": "multi-turn conversation",
    "general vqa": "general vqa",
    "text vqa": "text vqa",
    "image captioning": "image captioning",
}


def load_mmspec(n: int, per_category: int = 0) -> list[tuple[str, str, Path]]:
    """Return [(topic, question, image_path), ...]; download jsonl + referenced images.
    `per_category>0` takes K from every topic (capped at availability)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = CACHE / "mmspec.jsonl"
    if not meta.exists():
        logger.info("downloading MMSpec testmini metadata -> {}", meta)
        urllib.request.urlretrieve(f"{RAW}/mmspec.jsonl", meta)
    rows = [json.loads(line) for line in meta.read_text().splitlines()]
    if per_category:
        seen: Counter[str] = Counter()
        sel = []
        for rec in rows:
            topic = MMSPEC_TOPIC.get(rec["topic"], rec["topic"])
            if seen[topic] < per_category:
                sel.append(rec)
                seen[topic] += 1
        rows = sel
    else:
        rows = rows[:n]
    out = []
    for rec in tqdm(rows, desc="images", unit="img"):
        img = CACHE / "images" / rec["image"]
        if not img.exists():
            img.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{RAW}/images/{rec['image']}", img)
        topic = MMSPEC_TOPIC.get(rec["topic"], rec["topic"])
        out.append((topic, rec["turns"][0], img))
    return out


def main() -> None:
    p = argparse.ArgumentParser("dejavuu.eval.mmspec")
    p.add_argument("--dataset", choices=["mmspec", "specbench"], default="mmspec")
    p.add_argument("--methods", default="baseline,pld,token_recycling")
    p.add_argument("--workload", default="repetitive", help="specbench only")
    p.add_argument("--variant", choices=["q4", "int8"], default="q4")
    p.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--per-category", type=int, default=0, help="K per topic (overrides --n)")
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--budget", type=int, default=8)
    p.add_argument(
        "--tree",
        action="store_true",
        help="tree verify; needs a tree-capable decoder, else falls back to chain",
    )
    p.add_argument("--width", type=int, default=2, help="max children/node in tree mode")
    p.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="mmspec only: cap the "
        "processor's longest_edge (px) -> fewer 512px tiles -> cheaper "
        "image prefill. 0 = processor default",
    )
    p.add_argument(
        "--datastore",
        type=Path,
        default=None,
        help="corpus file (one doc per line) seeding REST / SAM-Decoding's static store",
    )
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--responses", type=Path, default=None, help="generated-response JSONL")
    p.add_argument("--out", type=Path, default=None, help="immutable structured run directory")
    p.add_argument(
        "--threads", type=int, default=0, help="ORT intra-op threads per session (0 = ORT default)"
    )
    args = p.parse_args()
    if args.out:
        if args.csv or args.log or args.responses:
            p.error(
                "--out owns summary/log/response paths; do not combine it with --csv/--log/--responses"
            )
        create_run_dir(
            args.out.parent,
            args.out.name,
            {
                "schema_version": 1,
                "dataset": args.dataset,
                "decode": {
                    "budget": args.budget,
                    "max_new": args.max_new,
                    "tree": args.tree,
                    "width": args.width,
                },
                "runtime": {"provider": args.provider, "threads": args.threads},
            },
        )
        args.csv = args.out / "summary.csv"
        args.log = args.out / "logs" / "runner.log"
        args.responses = args.out / "responses.jsonl"

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logger.add(args.log, mode="a", format="{time:HH:mm:ss} {level} {message}")

    methods = args.methods.split(",")
    if "baseline" in methods:  # must run first: every method's match/speedup is vs it
        methods = ["baseline", *(m for m in methods if m != "baseline")]
    spec_decoder = VLM_TREE_DECODER.exists() or GENAI_DECODER.exists()
    if any(m != "baseline" for m in methods) and not spec_decoder:
        logger.warning(
            "no spec-decode decoder ({} / {}) -- build with tools/build_tree_decoder.py "
            "(tree+hidden, needed for pld_plus/adapld) or tools/build_vlm_decoder.py "
            "(chain); only baseline will be correct.",
            VLM_TREE_DECODER,
            GENAI_DECODER,
        )

    from PIL import Image
    from transformers import AutoProcessor

    root = download(args.variant)
    vlm = VLM(Path(root), args.variant, args.provider, threads=args.threads)
    processor = AutoProcessor.from_pretrained(root)
    eos = processor.tokenizer.eos_token_id

    is_vlm = args.dataset == "mmspec"
    if is_vlm:
        prompts = load_mmspec(args.n, args.per_category)  # [(topic, question, img_path)]
    else:
        prompts = load_specbench(args.workload, args.n, args.per_category)  # [(topic, q)]
    datastore = load_datastore(args.datastore, processor.tokenizer) if args.datastore else None
    logger.info(
        "{} {} prompts | budget={} max_new={} image_size={}{}{} | methods={}",
        len(prompts),
        args.dataset,
        args.budget,
        args.max_new,
        args.image_size,
        f" tree(width={args.width})" if args.tree else "",
        f" datastore={len(datastore)}docs" if datastore else "",
        methods,
    )

    drafters = {m: make_drafter(m, datastore) for m in methods}
    cases = [
        RunCase(
            str(index), item[0], {"question": item[1], "image": str(item[2]) if is_vlm else None}
        )
        for index, item in enumerate(prompts)
    ]

    def prepare(case: RunCase, _: str) -> list[int]:
        # Build the chat template once per prompt; for VLM splice vision per method
        # (prepare re-stashes prefill embeds), for text reuse the same ids.
        question = str(case.metadata["question"])
        if is_vlm:
            img_path = Path(str(case.metadata["image"]))
            content = [{"type": "image"}, {"type": "text", "text": question}]
        else:
            content = [{"type": "text", "text": question}]
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True
        )
        # Two-step: templating with tokenize=True clashes with images=[img].
        if is_vlm:
            img = Image.open(img_path).convert("RGB")
            # images_kwargs is the form that actually applies (top-level size= warns
            # and processor_kwargs= is silently ignored on this processor).
            kw = (
                {"images_kwargs": {"size": {"longest_edge": args.image_size}}}
                if args.image_size
                else {}
            )
            proc_out = dict(processor(text=text, images=[img], return_tensors="np", **kw))
        else:
            text_ids = processor(text=text, return_tensors="np")["input_ids"][0].tolist()
        return vlm.prepare(proc_out) if is_vlm else text_ids

    aggs, responses, failures = run_cases(
        cases,
        methods,
        vlm,
        prepare,
        lambda method, _: drafters[method],
        lambda tokens: processor.tokenizer.decode(tokens, skip_special_tokens=True),
        max_new=args.max_new,
        budget=args.budget,
        eos=eos,
        tree=args.tree,
        width=args.width,
    )

    scope = f"{args.per_category}/topic" if args.per_category else args.workload
    render_table(
        f"SmolVLM / {args.dataset} / {scope} / {args.variant} (n={len(prompts)})",
        methods,
        aggs,
        args.log,
        strict=False,
        csv_path=args.csv,
    )
    if args.csv:
        write_car_profile(args.csv, aggs)
        write_response_jsonl(args.responses or args.csv.with_suffix(".responses.jsonl"), responses)
        write_response_jsonl(args.csv.with_suffix(".failures.jsonl"), failures)
        write_run_manifest(
            args.csv,
            benchmark_metadata(
                dataset=args.dataset,
                model=f"{root}:{args.variant}:{vlm.decoder_path}",
                provider=args.provider,
                threads=args.threads,
                budget=args.budget,
                tree=args.tree,
                width=args.width,
                max_new=args.max_new,
            ),
        )


if __name__ == "__main__":
    main()
