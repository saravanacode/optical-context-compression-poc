#!/usr/bin/env python3
"""Tier 1: token accounting for optical context compression.

Measures COST ONLY: how many decoder-sequence tokens does a document cost
as raw text vs. as rendered page images fed through a vision encoder.
This says nothing about whether the text can be read back — that's Tier 2.

    "Compression" here means fewer tokens in the sequence the LLM attends
    over. It is NOT information-theoretic compression (not gzip).
    Vision tokens are not free: a full encoder forward pass sits behind
    each one.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from render import measure_layout, paginate, render_text_to_pages, save_pages

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

# Vision-token arithmetic: tokens_per_page = (page_px / patch) ** 2 / pool.
# These profiles are DeepSeek-OCR's own published resolution modes
# (patch=16, 16x conv compressor) — arXiv:2510.18234, Table 1 / Sec 3.2.
# Different encoders pool differently; label the profile used, always.
ENCODER_PROFILES = {
    "deepseek-ocr-tiny":  {"page_px": 512,  "patch": 16, "pool": 16},
    "deepseek-ocr-small": {"page_px": 640,  "patch": 16, "pool": 16},
    "deepseek-ocr-base":  {"page_px": 1024, "patch": 16, "pool": 16},
    "deepseek-ocr-large": {"page_px": 1280, "patch": 16, "pool": 16},
}

CHARS_PER_TOKEN_FALLBACK = 4.0  # labeled APPROXIMATION when tiktoken unavailable


@dataclass
class TokenCount:
    n_tokens: int
    exact: bool


def count_text_tokens(text: str, encoding_name: str = "cl100k_base") -> TokenCount:
    try:
        import tiktoken

        enc = tiktoken.get_encoding(encoding_name)
        return TokenCount(n_tokens=len(enc.encode(text)), exact=True)
    except Exception as e:
        print(f"  [WARN] tiktoken unavailable ({e}); falling back to "
              f"~{CHARS_PER_TOKEN_FALLBACK} chars/token APPROXIMATION", file=sys.stderr)
        return TokenCount(n_tokens=int(len(text) / CHARS_PER_TOKEN_FALLBACK), exact=False)


def vision_tokens_for_profile(n_pages: int, profile: dict) -> int:
    tokens_per_page = (profile["page_px"] / profile["patch"]) ** 2 / profile["pool"]
    return int(round(tokens_per_page)) * n_pages


def run_sweep(
    doc_path: Path,
    font_sizes: list[int],
    profile_names: list[str],
    encoding_name: str,
) -> list[dict]:
    text = doc_path.read_text(errors="replace")
    tok = count_text_tokens(text, encoding_name)
    rows = []
    for font_size in font_sizes:
        # Page pixel size is fixed per profile; layout (cols/rows) depends
        # on font_size against that fixed canvas.
        for profile_name in profile_names:
            profile = ENCODER_PROFILES[profile_name]
            layout = measure_layout(profile["page_px"], font_size)
            pages = paginate(text, layout)
            n_pages = len(pages)
            v_tokens = vision_tokens_for_profile(n_pages, profile)
            ratio = tok.n_tokens / v_tokens if v_tokens else float("inf")
            rows.append(
                {
                    "document": doc_path.name,
                    "doc_chars": len(text),
                    "font_size": font_size,
                    "profile": profile_name,
                    "page_px": profile["page_px"],
                    "n_pages": n_pages,
                    "chars_per_page": layout.chars_per_page,
                    "text_tokens": tok.n_tokens,
                    "text_tokens_exact": tok.exact,
                    "vision_tokens": v_tokens,
                    "compression_ratio": round(ratio, 3),
                }
            )
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    exact_flag = "(exact)" if all(r["text_tokens_exact"] for r in rows) else "(APPROXIMATION)"
    print(f"\ntokenizer: cl100k_base {exact_flag}")
    header = f"{'document':<28} {'profile':<20} {'font':>5} {'pages':>6} {'text_tok':>9} {'vis_tok':>8} {'ratio':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['document']:<28} {r['profile']:<20} {r['font_size']:>5} {r['n_pages']:>6} "
            f"{r['text_tokens']:>9} {r['vision_tokens']:>8} {r['compression_ratio']:>7.2f}"
        )


def plot_curve(rows: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs = sorted({r["document"] for r in rows})
    profiles = sorted({r["profile"] for r in rows})

    fig, axes = plt.subplots(1, len(profiles), figsize=(6 * len(profiles), 5), squeeze=False)
    axes = axes[0]

    for ax, profile in zip(axes, profiles):
        for doc in docs:
            pts = sorted(
                (r["font_size"], r["compression_ratio"])
                for r in rows
                if r["document"] == doc and r["profile"] == profile
            )
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", label=doc)
        page_px = next(r["page_px"] for r in rows if r["profile"] == profile)
        ax.set_title(f"{profile}\n({page_px}px page)")
        ax.set_xlabel("font size (px)")
        ax.set_ylabel("compression ratio (text_tokens / vision_tokens)")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="break-even")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    exact = all(r["text_tokens_exact"] for r in rows)
    fig.suptitle(
        "Optical context compression: ratio is a dial (font size), not a property of the technique\n"
        f"tokenizer: cl100k_base {'(exact)' if exact else '(APPROXIMATION — tiktoken unavailable)'}"
        " | vision tokens = (page_px/patch)^2/pool, per DeepSeek-OCR published arithmetic"
        " | page-count is a step function, curve is jagged by design",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved chart: {out_path}")


def sanity_check_8px(doc_path: Path) -> None:
    """Render at 8px and save the PNG — the real answer to 'why not just
    use 6px'. If it's an unreadable smudge, that's the point."""
    text = doc_path.read_text(errors="replace")
    images, layout = render_text_to_pages(text, page_px=1024, font_size=8)
    out_dir = FIGURES_DIR / "sanity_8px"
    paths = save_pages(images[:1], out_dir, doc_path.stem)
    print(
        f"\n[sanity] {doc_path.name} @ 8px: {layout.cols} cols x {layout.rows} rows "
        f"({layout.chars_per_page} chars/page) -> {paths[0]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 1 token accounting sweep")
    parser.add_argument("--docs", nargs="+", type=Path, required=True, help="text documents to sweep")
    parser.add_argument(
        "--font-sizes", nargs="+", type=int, default=[8, 10, 12, 14, 16, 20, 24],
    )
    parser.add_argument(
        "--profile", default="deepseek-ocr-base",
        choices=list(ENCODER_PROFILES) + ["all"],
    )
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--sanity-8px", action="store_true", help="render one 8px sample page per doc")
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "tier1_sweep.csv")
    parser.add_argument("--out-chart", type=Path, default=FIGURES_DIR / "compression_chart.png")
    args = parser.parse_args()

    profile_names = list(ENCODER_PROFILES) if args.profile == "all" else [args.profile]

    all_rows: list[dict] = []
    for doc_path in args.docs:
        if not doc_path.exists():
            print(f"[ERROR] missing document: {doc_path}", file=sys.stderr)
            sys.exit(1)
        n_chars = len(doc_path.read_text(errors="replace"))
        if n_chars < 10_000:
            print(
                f"  [WARN] {doc_path.name} is only {n_chars} chars (<10k). "
                "Ratio curve will look flat — this is the 'short documents' gotcha, not a bug.",
                file=sys.stderr,
            )
        print(f"\n== {doc_path.name} ({n_chars} chars) ==")
        rows = run_sweep(doc_path, args.font_sizes, profile_names, args.encoding)
        all_rows.extend(rows)
        if args.sanity_8px:
            sanity_check_8px(doc_path)

    print_table(all_rows)
    write_csv(all_rows, args.out_csv)
    print(f"\nSaved table: {args.out_csv}")
    plot_curve(all_rows, args.out_chart)


if __name__ == "__main__":
    main()
