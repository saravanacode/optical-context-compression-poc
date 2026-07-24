# Optical Context Compression — POC Plan

## Goal

Measure, on my own machine, the real tradeoff in "optical context compression":
rendering text into an image and feeding it to a vision encoder, so the LLM
attends over fewer tokens than the text would cost.

Two deliverables:
1. A reproducible experiment with real numbers.
2. A chart + honest write-up I can post publicly without getting torn apart.

**The finding I expect to land on:** compression ratio is not a property of the
technique — it's a dial set by font size / text density. Every notch of
compression costs legibility. The interesting artifact is the *curve*, not a
single headline number.

---

## Ground rules (these matter more than the code)

Do not let the implementation drift into claims it can't support.

- "Compression" here means **fewer tokens in the sequence the LLM attends over**.
  It is NOT information-theoretic compression. Never compare it to gzip.
- **Vision tokens are not free.** A full encoder forward pass sits behind each
  one. The saving is decoder sequence length, which is what hurts as context grows.
- Vision-token counts are **derived from a specific encoder's published
  patch/pool arithmetic**. Different models pool differently. Always label the
  profile used.
- Tier 1 measures cost only. It says **nothing** about whether text can be read
  back. Do not imply recovery accuracy until Tier 2 exists.
- If a number came from an approximation rather than a real tokenizer, the
  output must say so explicitly.

---

## Milestone 1 — Tier 1: token accounting (target: one afternoon)

**Status: already built.** Starting script: `optical_compression_poc.py`.
Read it first, then extend rather than rewrite.

What it does:
- Counts text tokens with `tiktoken` (falls back to ~4 chars/token, labeled).
- Renders text to square page images with PIL, paginating as needed.
- Computes vision tokens as `(res/patch)^2 / pool` per page, times page count.
- Sweeps font size and plots the tradeoff curve.

### Tasks to finish M1

- [ ] Verify `tiktoken` loads its vocab. If output says `APPROXIMATION`, the
      vocab download was blocked — fix the network or vendor the vocab file.
      Numbers are not publication-grade until this says `(exact)`.
- [ ] Run against **3 real documents**, not the built-in sample:
      a paper's body text, a long README, and a source file.
      Each should be 10k+ chars or the curve goes flat.
- [ ] Add a `--profile all` mode that tabulates every encoder profile at once.
- [ ] Sanity-check the page-capacity math: render at 8px and actually *look* at
      the PNG. If it's an unreadable smudge, that's the real answer to
      "why not just use 6px" — capture that image, it's good post material.

**Acceptance:** a table of (document, font size, text tokens, vision tokens,
ratio) with the exact-tokenizer flag set, plus `compression_chart.png`.

---

## Milestone 2 — Tier 2: the round-trip (target: a weekend)

This is the one that earns respect, because it closes the obvious hole: *can you
actually get the text back?*

### Setup

- Model: `deepseek-ai/DeepSeek-OCR` on Hugging Face.
- **Check the model card for the current inference API before writing code.**
  Do not guess method signatures from memory — read the card, run their example
  first, confirm it works, then build around it.
- Needs a GPU. Confirm VRAM headroom before committing to the larger modes.
- Note which resolution mode you invoke; it determines the vision-token count
  and must match what Tier 1 assumed.

### Tasks

- [ ] Get the vendor's example running unmodified. Do not proceed until this works.
- [ ] Build the pipeline: text → rendered PNG → model → recovered text.
- [ ] Add accuracy metrics. Use `jiwer` (or `editdistance`) for:
      - CER (character error rate) — the primary metric
      - WER (word error rate) — secondary, more intuitive for a caption
- [ ] Normalize before comparing: collapse whitespace, decide on a case policy,
      and **document the normalization**. Sloppy normalization is the easiest way
      to fake a good CER, and the easiest thing for a reader to call out.
- [ ] Sweep font size (and/or resolution mode) and record CER at each point.

**Acceptance:** a single plot with compression ratio on X and CER on Y. That
plot is the whole project. Everything else is scaffolding.

---

## Milestone 3 — Tier 3: find the cliff

The honest flex. A demo that only shows the win is marketing; a demo that shows
where it breaks is engineering.

- [ ] Build a small corpus of deliberately adversarial inputs:
      - clean running prose (the easy baseline)
      - source code with meaningful indentation
      - a dense numeric table
      - a two-column layout
      - small-font / low-contrast text
- [ ] Run each through the Tier 2 pipeline at matched compression ratios.
- [ ] Plot CER per document type on shared axes.

**Expected result:** prose degrades gracefully; code and tables fall off a cliff
much earlier. If that's what I find, it's a real, publishable observation and it
lines up with the existing skeptical literature — there's a preprint arguing
optical context compression underperforms simpler methods at longer scales.
Read it before writing the post so I'm engaging with the counterargument rather
than getting ambushed by it.

---

## Repo layout

```
optical-compression-poc/
├── plan.md
├── README.md              # written LAST, from actual results
├── requirements.txt
├── src/
│   ├── tier1_tokens.py    # existing script, extended
│   ├── tier2_roundtrip.py
│   └── render.py          # shared text→image rendering
├── data/                  # test documents (3+ real ones, 10k+ chars)
├── results/               # CSVs — commit these, they're the evidence
└── figures/               # charts for the post
```

Commit the results CSVs. If someone asks "what were your actual numbers," the
answer should be a link, not a screenshot.

---

## Known gotchas

- **Short documents make the effect vanish.** Under ~1 page, the ratio is
  dominated by fixed page overhead and the curve is flat. 10k+ chars minimum.
- **Monospace fonts only** for the capacity math. The existing script assumes a
  fixed character width; a proportional font silently breaks the estimate.
- **Page count is a step function.** Ratio jumps at each page boundary, so the
  curve looks jagged. That's real, not a bug — don't smooth it away.
- **Resolution mode must match between Tier 1 and Tier 2.** If Tier 1 assumed
  1024×1024/256 tokens and Tier 2 runs a different mode, the ratios aren't
  comparable and the whole comparison is invalid.
- **Don't cherry-pick the font size** that gives the prettiest number. Report the
  sweep.

---

## Publishing checklist

Before anything goes public:

- [ ] Tokenizer says `(exact)`, not `APPROXIMATION`
- [ ] Every chart states the encoder profile and resolution used
- [ ] Caption clarifies "compression" = LLM sequence length, not gzip
- [ ] Caption notes vision tokens carry an encoder forward pass
- [ ] If only Tier 1 is done, say plainly that reconstruction was not measured
- [ ] Normalization method stated wherever CER appears
- [ ] The failure cases are shown, not just the wins
- [ ] Repo link with results CSVs committed

---

## Sources to read before writing anything public

- DeepSeek-OCR technical report — arxiv.org/pdf/2510.18234
  (the "contexts optical compression" proof-of-concept; its ratios are the
  authors' own reported results, cite them as such)
- Vision-centric Token Compression (Vist), NeurIPS 2025 —
  openreview.net/forum?id=YdggdEL41C
- "Optical Context Compression Is Just (Bad) Autoencoding" —
  arxiv.org/pdf/2512.03643 (the counterargument; read it properly)

---

## Stop condition

Tier 1 + a real chart is already a publishable result. Ship that first. Tier 2
is a follow-up post, not a prerequisite — and shipping twice from one idea beats
sitting on it for three weeks chasing a CUDA install.