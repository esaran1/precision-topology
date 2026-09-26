# Track C: verified positioning — writer inputs

Verified 2026-09-26 (about 01:15–02:00 EDT). Every field was read from the official page for its venue: proceedings.neurips.cc,
proceedings.mlr.press (COLT), proceedings.iclr.cc (ICLR 2024), iclr.cc (ICLR 2022, since proceedings.iclr.cc has no 2022
volume), and the DOI record (Fenichel). Each was then cross-checked against arXiv where an arXiv version exists. The
conventions are those of WP-13 and WP-26:
- fields come from the official page's own BibTeX;
- `editor` and `url` are left out, and the URL goes in a `% verified:` comment line;
- ICLR's `volume = {<year>}` is left out;
- author forms follow the official BibTeX.

Each relation sentence is based on the full official abstract. Fenichel is the exception: no abstract is available (see
item 10). Nothing is quoted.

None of the ten papers is in `census/paper/references.bib`. The new keys do not collide with existing ones, but note
that `lyu2021gradient` (NeurIPS 2021) sits beside the existing `lyu2020gradient` (ICLR 2020, a different paper).

**Result: 10 INCLUDED, 0 EXCLUDED.** One item needs the coordinator's check: Fenichel (item 10) was verified against
the DOI registration record, because ScienceDirect refused automated access.

## 1. Verification table

| # | key | venue | official URL | arXiv | differences to note |
|---|---|---|---|---|---|
| 1 | `rosset2003margin` | NIPS 2003 (NeurIPS vol. 16), MIT Press | https://proceedings.neurips.cc/paper/2003/hash/0fe473396242072e84af286632d3f0ff-Abstract.html | none | The page displays "Trevor J. Hastie"; the official BibTeX gives "Hastie, Trevor" (used). The official BibTeX has empty `pages`, so the field is omitted. **Choice:** the brief describes norm-constrained logistic minimisers converging to max-margin as the norm grows. This paper proves that regularised-loss solutions (logistic included) converge to margin-maximising separators as the regularisation vanishes. It is the only NIPS 2003 match. The same authors' "Boosting as a regularized path to a maximum margin classifier" is JMLR 2004, not NeurIPS 2003. |
| 2 | `ji2020gradient` | COLT 2020, PMLR 125:2109–2136 | https://proceedings.mlr.press/v125/ji20a.html | 2006.11226 ("To appear, COLT 2020") | none (the title, authors and abstract are identical). |
| 3 | `moroshko2020implicit` | NeurIPS 2020 (vol. 33), pp. 22182–22193 | https://proceedings.neurips.cc/paper/2020/hash/fc2022c89b61c76bbef978f1370660bf-Abstract.html | 2007.06738 | **Author order differs:** the proceedings give Moroshko, Woodworth, Gunasekar, Lee, Srebro, Soudry, while arXiv swaps Woodworth and Gunasekar. arXiv has "Nathan Srebro" where the proceedings have "Nati Srebro". The page displays "Jason Lee"; the official BibTeX has "Lee, Jason D" (used). The entry uses the proceedings order. The NeurIPS BibTeX gives no DOI for 2020. |
| 4 | `kunin2025alternating` | NeurIPS 2025 (vol. 38, Main Conference Track), pp. 4377–4424, DOI 10.52202/085713-0156 | https://proceedings.neurips.cc/paper_files/paper/2025/hash/06cbd2e81dfbd3bb4cb0abce95b32584-Abstract-Conference.html | 2506.06489 ("NeurIPS 2025") | **The NeurIPS 2025 proceedings are online** (checked in the vol38-main-conference listing), so no OpenReview fallback was needed. The page displays "James Simon" and "Michael R. Deweese"; the official BibTeX has "Simon, James" and "Deweese, Michael" (used). arXiv has "James B. Simon" and "Michael R. DeWeese" (capital W). The official BibTeX's `volume = {38, Main Conference}` is written as `volume = {38}`. The official abstract has one clause that arXiv lacks (that all neurons dormant corresponds to initialisation at the origin). |
| 5 | `morwani2024feature` | ICLR 2024, pp. 29077–29114 | https://proceedings.iclr.cc/paper_files/paper/2024/hash/7d4647db960780c8b5a4e7d9e4a58d68-Abstract-Conference.html | 2311.07568 (arXiv comment: "Accepted as Spotlight at ICLR 2024") | arXiv has "Benjamin L. Edelman"; the proceedings have "Edelman, Benjamin" (used). Per WP-13, do not cite the "Spotlight" label. |
| 6 | `lyu2021gradient` | NeurIPS 2021 (vol. 34), pp. 12978–12991 | https://proceedings.neurips.cc/paper/2021/hash/6c351da15b5e8a743a21ee96a86e25df-Abstract.html | 2110.13905 ("Published in NeurIPS 2021") | none (arXiv's metadata order is Lyu, Li, Wang, Arora, the same as the proceedings; the abstract is identical). |
| 7 | `atanasov2022neural` | ICLR 2022 | https://iclr.cc/virtual/2022/poster/7005 | 2111.00034 (journal ref "ICLR 2022") | **proceedings.iclr.cc has no 2022 volume** (page not found). openreview.net (forum 1NvflqAdoom) returned a challenge (403) to automated requests. So iclr.cc is the source, the same fallback WP-13 used for ICLR 2023. There is no volume, so there are no pages. The title, authors and abstract are identical on iclr.cc and arXiv. |
| 8 | `kalimeris2019sgd` | NeurIPS 2019 (vol. 32) | https://proceedings.neurips.cc/paper/2019/hash/b432f34c5a997c8e7c806a895ecc5e25-Abstract.html | 1905.11604 ("Submitted to NeurIPS 2019") | **The first author differs.** The proceedings list Kalimeris, Kaplun, Nakkiran, Edelman, Yang, Barak, Zhang, while arXiv lists Nakkiran, Kaplun, Kalimeris, Yang, Edelman, Zhang, Barak. The proceedings give "Haofeng Zhang" where arXiv gives "Fred Zhang", and "Benjamin Edelman" where arXiv gives "Benjamin L. Edelman". The key and the in-text citation follow the proceedings, so it is **Kalimeris et al. (2019)**, not "Nakkiran et al." (Lyu et al. 2021 also cite it as Kalimeris et al.). The official BibTeX has empty `pages`, so the field is omitted. |
| 9 | `shah2020pitfalls` | NeurIPS 2020 (vol. 33), pp. 9573–9585 | https://proceedings.neurips.cc/paper/2020/hash/6cfe0e6127fa25df2a0ef2ae1067d915-Abstract.html | 2006.07710 ("NeurIPS 2020") | **The abstracts differ.** The proceedings abstract cites different works in its first sentences, says "introducing piecewise-linear and image-based datasets" where arXiv says "designing datasets", and differs in minor wording. The sentence below uses only content found in both. |
| 10 | `fenichel1979geometric` | J. Differential Equations 31(1):53–98, Jan 1979 (journal article) | https://doi.org/10.1016/0022-0396(79)90152-9 → https://www.sciencedirect.com/science/article/pii/0022039679901529 | none | **Coordinator check.** ScienceDirect (the article page and the vol. 31 issue page) returned a Cloudflare 1000-series error to automated access, so the HTML article page could not be read. The fields are from the DOI's registered record: doi.org content negotiation and Crossref, publisher Elsevier BV, type journal-article. That record gives Fenichel, Neil; the title; the Journal of Differential Equations; vol. 31, no. 1, pp. 53–98, January 1979; ISSN 0022-0396. The DOI resolves (through linkinghub.elsevier.com) to the ScienceDirect PII above. The record has no abstract, and the 1979 article has none on record, so the sentence below rests on the title and on the paper's standard content (see item 10). |

## 2. BibTeX

```bibtex
% verified: https://proceedings.neurips.cc/paper/2003/hash/0fe473396242072e84af286632d3f0ff-Abstract.html
@inproceedings{rosset2003margin,
  title     = {Margin Maximizing Loss Functions},
  author    = {Rosset, Saharon and Zhu, Ji and Hastie, Trevor},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {16},
  publisher = {MIT Press},
  year      = {2003}
}

% verified: https://proceedings.mlr.press/v125/ji20a.html
@inproceedings{ji2020gradient,
  title     = {Gradient descent follows the regularization path for general losses},
  author    = {Ji, Ziwei and Dud{\'i}k, Miroslav and Schapire, Robert E. and Telgarsky, Matus},
  booktitle = {Proceedings of Thirty Third Conference on Learning Theory},
  series    = {Proceedings of Machine Learning Research},
  volume    = {125},
  pages     = {2109--2136},
  publisher = {PMLR},
  year      = {2020}
}

% verified: https://proceedings.neurips.cc/paper/2020/hash/fc2022c89b61c76bbef978f1370660bf-Abstract.html
@inproceedings{moroshko2020implicit,
  title     = {Implicit Bias in Deep Linear Classification: Initialization Scale vs Training Accuracy},
  author    = {Moroshko, Edward and Woodworth, Blake E and Gunasekar, Suriya and Lee, Jason D and Srebro, Nati and Soudry, Daniel},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  pages     = {22182--22193},
  publisher = {Curran Associates, Inc.},
  year      = {2020}
}

% verified: https://proceedings.neurips.cc/paper_files/paper/2025/hash/06cbd2e81dfbd3bb4cb0abce95b32584-Abstract-Conference.html
@inproceedings{kunin2025alternating,
  title     = {Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks},
  author    = {Kunin, Daniel and Marchetti, Giovanni Luca and Chen, Feng and Karkada, Dhruva and Simon, James and Deweese, Michael and Ganguli, Surya and Miolane, Nina},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {38},
  pages     = {4377--4424},
  publisher = {Curran Associates, Inc.},
  doi       = {10.52202/085713-0156},
  year      = {2025}
}

% verified: https://proceedings.iclr.cc/paper_files/paper/2024/hash/7d4647db960780c8b5a4e7d9e4a58d68-Abstract-Conference.html
@inproceedings{morwani2024feature,
  title     = {Feature emergence via margin maximization: case studies in algebraic tasks},
  author    = {Morwani, Depen and Edelman, Benjamin and Oncescu, Costin-Andrei and Zhao, Rosie and Kakade, Sham},
  booktitle = {International Conference on Learning Representations},
  pages     = {29077--29114},
  year      = {2024}
}

% verified: https://proceedings.neurips.cc/paper/2021/hash/6c351da15b5e8a743a21ee96a86e25df-Abstract.html
@inproceedings{lyu2021gradient,
  title     = {Gradient Descent on Two-layer Nets: Margin Maximization and Simplicity Bias},
  author    = {Lyu, Kaifeng and Li, Zhiyuan and Wang, Runzhe and Arora, Sanjeev},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {34},
  pages     = {12978--12991},
  publisher = {Curran Associates, Inc.},
  year      = {2021}
}

% verified: https://iclr.cc/virtual/2022/poster/7005 (proceedings.iclr.cc has no 2022 volume; openreview.net refused automated access; arXiv 2111.00034 agrees)
@inproceedings{atanasov2022neural,
  title     = {Neural Networks as Kernel Learners: The Silent Alignment Effect},
  author    = {Atanasov, Alexander and Bordelon, Blake and Pehlevan, Cengiz},
  booktitle = {International Conference on Learning Representations},
  year      = {2022}
}

% verified: https://proceedings.neurips.cc/paper/2019/hash/b432f34c5a997c8e7c806a895ecc5e25-Abstract.html
@inproceedings{kalimeris2019sgd,
  title     = {{SGD} on Neural Networks Learns Functions of Increasing Complexity},
  author    = {Kalimeris, Dimitris and Kaplun, Gal and Nakkiran, Preetum and Edelman, Benjamin and Yang, Tristan and Barak, Boaz and Zhang, Haofeng},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {32},
  publisher = {Curran Associates, Inc.},
  year      = {2019}
}

% verified: https://proceedings.neurips.cc/paper/2020/hash/6cfe0e6127fa25df2a0ef2ae1067d915-Abstract.html
@inproceedings{shah2020pitfalls,
  title     = {The Pitfalls of Simplicity Bias in Neural Networks},
  author    = {Shah, Harshay and Tamuly, Kaustav and Raghunathan, Aditi and Jain, Prateek and Netrapalli, Praneeth},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  pages     = {9573--9585},
  publisher = {Curran Associates, Inc.},
  year      = {2020}
}

% verified: https://doi.org/10.1016/0022-0396(79)90152-9 (DOI registration record, publisher Elsevier BV; resolves to https://www.sciencedirect.com/science/article/pii/0022039679901529, which refused automated access)
@article{fenichel1979geometric,
  title     = {Geometric singular perturbation theory for ordinary differential equations},
  author    = {Fenichel, Neil},
  journal   = {Journal of Differential Equations},
  volume    = {31},
  number    = {1},
  pages     = {53--98},
  doi       = {10.1016/0022-0396(79)90152-9},
  year      = {1979}
}
```

Consistency note: `references.bib` writes some shared authors differently (for example "Jason D. Lee" in `jin2023understanding`).
The entries above use each page's official form, as WP-13 and WP-26 did. Choose one form for the whole bibliography.

## 3. One sentence per paper (shared / new)

Each sentence is paraphrased from the full official abstract and quotes nothing. Terms used below:
- **Placed / unplaced:** the two minimiser types of the loss at fixed output scale s.
- **R_glob:** the certified switch.
- **r = κ(a)χ:** the lag.

1. **Rosset, Zhu and Hastie (2003).** They prove a sufficient condition, covering the hinge, exponential and logistic
   losses, under which minimisers of a regularised loss converge to a margin-maximising separator as the
   regularisation vanishes.
   - Shared: we study the same kind of regularization path, the minimiser at fixed output scale as the scale grows,
     and its large-scale end separates every point.
   - New: for a non-homogeneous, non-monotone unit, the path is not a smooth approach to the margin solution. It jumps
     at a finite, interval-certified scale R_glob from an unplaced minimiser (driven by the class-mean gap) to a placed
     one.
2. **Ji, Dudík, Schapire and Telgarsky (2020).** For linear predictors with any convex, strictly decreasing loss whose
   risk does not attain its infimum, they show that the gradient-descent path and the algorithm-independent
   regularization path converge to the same direction, which is the maximum-margin direction for exponentially tailed
   losses.
   - Shared: we also read training against the regularization path.
   - New: our model is nonlinear and non-homogeneous, and its path has a certified discontinuity at finite scale.
     Training is related to that path by a derived, parameter-free lag r = κ(a)χ at the switch, not by agreement of
     asymptotic directions.
3. **Moroshko et al. (2020).** For gradient flow on the exponential loss over diagonal linear networks, they show that
   the transition between kernel and rich regimes is controlled by how the initialisation scale relates to training
   accuracy. The asymptotic limits can need extreme training accuracies, and at realistic scales the implicit bias is
   more complex than those limits.
   - Shared: a scale variable decides which solution applies, and limit results do not describe behaviour at finite
     scale.
   - New: our scale variable is the output scale along the fixed-scale path of a single non-monotone unit, where the
     change of solution is a certified jump at a finite threshold rather than a gradual transition, and training
     reaches it with a quantified lag.
4. **Kunin et al. (2025).** Alternating Gradient Flows describes feature learning in two-layer networks trained from
   small initialisation. It alternates maximising a utility over dormant neurons with minimising a cost over active
   ones, which predicts the order, timing and size of the loss drops.
   - Shared: an abrupt acquisition of a feature is explained by a reduced account in which the rest of the dynamics
     follows an optimum.
   - New: our switch is between two minimisers of the loss at fixed output scale for one unit, located by interval
     arithmetic, and its timing in training is predicted by a first-order tracking lag and tested for Adam and SGD
     rather than derived in a small-initialisation limit of gradient flow.
5. **Morwani et al. (2024).** For modular addition, sparse parities and finite group operations, they prove that
   margin maximisation alone specifies the features learned by stylised networks: Fourier features and irreducible
   representations.
   - Shared: we use the large-scale, margin-type solution to say which feature a unit adopts.
   - New: we follow the whole scale path, not only its limit. At finite scale the preferred feature changes
     discontinuously, from the one set by the class-mean gap to the placed one, at a certified threshold.
6. **Lyu, Li, Wang and Arora (2021).** For two-layer Leaky ReLU networks trained by gradient flow on linearly separable,
   symmetric data, they prove global margin optimality at any width. They also justify the early simplicity bias of
   gradient descent toward linear solutions, and show that a simple data change can send gradient flow to a
   suboptimal-margin linear classifier.
   - Shared: a simple solution comes before a margin-type one.
   - New: outside the homogeneous setting (a non-monotone activation), the change from the simple, class-mean-gap
     solution to the placed one is a certified discontinuity of the fixed-scale minimiser, and its timing in training
     is predicted.
7. **Atanasov, Bordelon and Pehlevan (2022).** In homogeneous networks with small initialisation and whitened data, the
   tangent kernel changes its eigenstructure while small, before the loss falls appreciably, and afterwards only grows
   in scale (silent alignment). They treat the linear case analytically.
   - Shared: the dynamics split into a direction or placement variable and an overall scale that grows.
   - New: in our non-homogeneous unit the order is reversed. Placement is not settled while the scale is small; it is
     triggered by scale growth, at a certified threshold plus the tracking lag.
8. **Kalimeris et al. (2019).** Experimentally, the early performance of SGD-trained networks is almost all explained by
   a linear classifier, and SGD learns functions of increasing complexity while keeping that early linear classifier.
   The measure used is based on conditional mutual information, with a theoretical result in a simplified model.
   - Shared: a simpler solution first, a more complex one later.
   - New: for one unit, we identify what sets the change: the fixed-scale minimiser switches discontinuously at a
     certified output scale. We also predict quantitatively when training makes the switch.
9. **Shah et al. (2020).** On datasets with a precise notion of simplicity, they show that simplicity bias can be
   extreme, with networks relying only on the simplest feature. This can explain non-robustness to small shifts and
   perturbations, can hurt generalisation even on the same distribution, and is not removed by ensembles or
   adversarial training.
   - Shared: training can prefer a simple solution over one that separates the data better.
   - New: in our model, preferring the unplaced (class-mean-gap) solution is a property of the loss at small output
     scale that ends at a certified threshold. So whether training leaves the simple solution is set by output-scale
     growth, with a predicted lag.
10. **Fenichel (1979).** This is the founding paper of geometric singular perturbation theory for ordinary differential
    equations. It proves that normally hyperbolic slow (invariant) manifolds persist in fast–slow systems, and it is the
    basis for slow-manifold tracking.
    - Shared: our lag r = κ(a)χ is a first-order slow-manifold tracking calculation of this kind, and the concept is not
      new.
    - New: we apply it to a discrete optimiser: κ(a) is computed from the landscape with no fitted parameter, the lag at
      the placement switch is tested quantitatively for Adam (and SGD), and we identify where it stops holding.

    *Coordinator check:* no official abstract was available (see the table). This sentence rests on the title and on
    the paper's standard, widely cited content. Adam and SGD are discrete, preconditioned maps outside the theorem's
    continuous-time hypotheses, so cite Fenichel for the concept only, not as a guarantee.

## 4. Novelty paragraph (writer wording, to the author's four points)

Tracking a slow manifold is not new. It goes back to geometric singular perturbation theory (Fenichel, 1979). The
relation between training and the regularization path (Rosset et al., 2003; Ji et al., 2020), the role of scale in
the kernel-to-rich transition (Moroshko et al., 2020), margin-based accounts of learned features (Morwani et al., 2024;
Lyu et al., 2021), simplicity bias (Kalimeris et al., 2019; Shah et al., 2020), and reduced descriptions of abrupt
feature acquisition (Atanasov et al., 2022; Kunin et al., 2025) are all established. What is new here is four things:
1. A non-homogeneous, non-monotone setting in which the fixed-scale (regularization) path has a certified placement
   discontinuity.
2. The first-order slow-manifold lag holding quantitatively for Adam.
3. The registered learning-rate invariance.
4. The identified validity boundary (χ ≲ 0.06).

Precision notes for the writer. These come from existing writer inputs, not from Track C.
- **"Holding quantitatively for Adam" (point 2).** In WP-24, the first-order κ(a)χ predicts all 36 arms within
  tolerance, with the observed slope within 30% of κ(a) (0.82–0.91, derived after a fitted relationship was known).
  The exact linear response (WP-31, POST HOC, predictions hashed before comparison) gives 1.00–1.05 per arm. Keep those
  labels wherever point 2 is stated.
- **"The registered learning-rate invariance" (point 3)** is R4 in WP-24: free Adam at η = 0.01 / 0.005 / 0.0025, PASS at
  a = 1.30 and 1.50.
- **"The identified validity boundary" (point 4)** is χ ≲ 0.06, with the run relaxed onto a branch. It was identified
  from data (Track 2) and not registered.

## 5. Do not say

- Do not claim novelty for slow-manifold tracking, adiabatic following, or first-order lag analysis as such. Cite
  Fenichel (1979) for the concept.
- Do not say Fenichel's theorems guarantee tracking for Adam or SGD. They concern continuous-time ODEs; our evidence for
  discrete optimisers is empirical, with κ derived.
- Do not claim novelty for results the cited papers establish:
  - the regularization path converging to the maximum-margin separator (Rosset et al.);
  - gradient descent following the regularization path in direction (Ji et al.);
  - scale-controlled kernel-to-rich transitions (Moroshko et al.);
  - margin maximisation determining features (Morwani et al.; Lyu et al.);
  - simplicity bias or increasing complexity (Kalimeris et al.; Shah et al.);
  - silent alignment (Atanasov et al.);
  - staircase or alternating accounts of feature learning (Kunin et al., 2025).
- Do not say any of these papers studies a certified discontinuity of the fixed-scale path, or a threshold in output
  scale (the same rule as WP-13 and WP-26).
- Do not call the lag law validated outside its boundary. It fails outside the sine family (GELU/SiLU/Mish), at width 2
  at full speed, and above χ ≈ 0.06.
- Do not cite "Nakkiran et al. (2019)". The proceedings list Kalimeris first.
- Do not cite presentation labels (for example "Spotlight" for Morwani et al.), per WP-13.
- Do not quote any abstract. Every sentence above is a paraphrase.
