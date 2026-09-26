# Four citations: verification (for the ICLR 2027 submission)

Verified 2026-09-26. Each field was read from the official page for its venue and then cross-checked against arXiv
(arxiv.org/abs pages, same day). Conventions as in WP-13, WP-26 and `results/track_c_writer_inputs.md`: fields from
the official page's own BibTeX where one exists; `editor` and `url` omitted, with the URL on a `% verified:` comment
line; ICLR's `volume = {<year>}` omitted; author forms follow the official BibTeX. BibTeX: `results/four_citations.bib`.

**Result: 4 INCLUDED, 0 EXCLUDED.** Each description matched exactly one paper at the stated venue and year (checked
against the full PMLR v267 / v235 listings, the iclr.cc 2023 paper list and the proceedings.iclr.cc 2025 listing), so no
description was ambiguous.

**Keys.** None of the four papers is in `census/paper/references.bib` or in any existing writer-input file, and the keys
do not collide with any existing key. Neighbours to keep apart: `kunin2023asymmetric` sits beside `kunin2024getrich`
and `kunin2025alternating`; `atanasov2025optimization` beside `atanasov2022neural`; `cai2025implicit` beside
`cai2023achieve` (a different Cai and a different paper).

## Verification table

| # | key | venue | official URL | arXiv | differences to note |
|---|---|---|---|---|---|
| 1 | `kunin2023asymmetric` | ICLR 2023 | https://iclr.cc/virtual/2023/poster/10967 | 2210.03820 (comment "ICLR 2023") | **proceedings.iclr.cc has no 2023 volume** (404), and openreview.net (forum IM4xp7kGI5V) returned a challenge (403) to automated requests, so iclr.cc is the source, the same fallback used for `atanasov2022neural`. iclr.cc gives no BibTeX, so the entry has only title, authors, booktitle and year; with no volume there are no pages. Title, authors (Kunin, Yamamura, Ma, Ganguli) and abstract are identical on iclr.cc and arXiv. iclr.cc labels it "top 25% paper"; per WP-13, do not cite presentation labels. |
| 2 | `cai2025implicit` | ICML 2025, PMLR 267:6426–6504 | https://proceedings.mlr.press/v267/cai25m.html | 2502.16075 (journal ref "ICML 2025") | arXiv has "Peter L. Bartlett"; the PMLR BibTeX has "Bartlett, Peter" (used). Abstracts agree except typography (PMLR writes "Ji & Telgarsky (2020)" with a missing space before it; arXiv "Ji and Telgarsky (2020)"). |
| 3 | `tsoy2024simplicity` | ICML 2024, PMLR 235:48728–48767 | https://proceedings.mlr.press/v235/tsoy24a.html | 2405.17299 (comment "ICML 2024, camera-ready version") | none (title, authors and abstract are identical). |
| 4 | `atanasov2025optimization` | ICLR 2025, pp. 5740–5779 | https://proceedings.iclr.cc/paper_files/paper/2025/hash/112c37467b801ad82d5e9c2ea4ff73f2-Abstract-Conference.html | 2410.04642 (comment "ICLR 2025 Final Copy") | arXiv has "James B. Simon"; the proceedings have "Simon, James" (used; the same form as in `kunin2025alternating`). **The page's "Bibtex" link is broken** (it points to `/paper_files/paper/2465-/bibtex`, which returns nothing); the BibTeX was read from the proceedings' own file `.../2025/file/112c37467b801ad82d5e9c2ea4ff73f2-Bibtex-Conference.bib`, which gives the pages, `volume = {2025}` (omitted) and no publisher. `{SGD}` is brace-protected in the title (as in `kalimeris2019sgd`); the official BibTeX has no braces. Abstracts identical apart from spacing and quote marks. |

## What each paper studies (from the official abstract; paraphrased, nothing quoted)

1. `kunin2023asymmetric`: extends maximum-margin results for homogeneous networks to quasi-homogeneous models (covering
   biases, residual connections, normalisation) trained by gradient flow on exponential loss after separability, and shows
   the implied margin is measured by an asymmetric norm that favours some parameters, with consequences for robustness,
   sparsity and Neural Collapse.
2. `cai2025implicit`: proves, for generic non-homogeneous deep networks under exponential loss once the empirical risk is
   small enough, that the normalised margin increases nearly monotonically, the gradient-descent iterates converge in
   direction, and the limit direction satisfies the KKT conditions of a margin-maximisation problem.
3. `tsoy2024simplicity`: characterises simplicity bias of small-initialisation two-layer networks under gradient flow for
   general (not linearly separable) data: early features cluster around a few width-independent directions, and for
   XOR-like data the bias strengthens later in training.
4. `atanasov2025optimization`: an empirical (plus simple-model) study of online SGD as the output down-scaling γ that sets
   feature-learning strength varies: the optimal learning rate scales as γ² for small γ and γ^(2/L) for large γ, and
   the large-γ ("ultra-rich") regime shows plateau-then-drop loss curves, trajectories equal up to a time
   reparameterisation, and often the best online performance.
