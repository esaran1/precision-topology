# Part B: related-work citations, verified (for the ICLR rebuttal revision)

Checked 2026-11-14 against primary pages only (PMLR, proceedings.neurips.cc, proceedings.iclr.cc, iclr.cc,
jmlr.org, arXiv abs, Crossref DOI records for journals, the BMVA archive for BMVC). No PDFs were downloaded.
DBLP returned a bot challenge, so it was not used. openreview.net was not needed.

**Scope.** The submitted PDF (`Precision_Topology_ICLR (18).pdf`) cites 61 works, all in §7 or Appendix T.
That is more than `paper/references.bib`, which has only the 33 older entries. The other 28 entries come from
`results/references_corrected.bib` and `paper/WRITER_INPUTS_v4_patch.md`. I re-checked every one of the 61
against its primary page, and also checked the page ranges and DOIs against the proceedings' own BibTeX
files. The comparison is with the entries as **printed in the submitted PDF bibliography**.

**Counts.** 60 verified, 1 corrected (`ren2026low`) and 8 new.

**Metadata errors in the submission: none.** Every author list, title, venue and year matches its primary page.
`ren2026low` is incomplete rather than wrong. When the paper was submitted, the ICML 2026 PMLR volume was not
online. It is now online as PMLR 306 (published 29 Sep 2026), and the paper appears at pp. 104476–104499. Add the
volume and pages. The variants below are only differences in how the primary pages render names. None is an
error, and the PDF uses the proceedings' BibTeX form in each case:
- Garipov: the NeurIPS page shows "Dmitry P Vetrov, Andrew G Wilson".
- Rosset: the page shows "Trevor J. Hastie".
- Moroshko: the abstract page shows "Jason Lee"; the BibTeX has "Jason D Lee", which is the form the PDF uses.
- Kunin 2025: the page text shows "Michael R. Deweese"; the BibTeX has "Deweese, Michael".
- Telgarsky: PMLR prints the title in lower case.
- Lyu 2024: "Simon Du, Jason Lee" as in the proceedings.
- Atanasov 2025 / Kunin 2025: "James Simon" (arXiv has "James B. Simon").
- Jacot et al. 2021 is still arXiv-only. Its arXiv page has no journal-ref (v2, 31 Jan 2022).

**Themes the brief suggested that are already cited:** Chizat et al. 2019, Kumar et al. 2024, Lyu et al. 2024,
Jacot et al. 2021, Pesme & Flammarion 2023. None of these is new.
**Not added:** central flows / further edge-of-stability work. The paper already cites Cohen 2021, Ahn 2022,
Zhu 2023 and Ahn 2023 for the EoS question. A central-flow reference would matter only if a reviewer attributes
the lag to EoS oscillation.

Status key: **verified** = the printed entry matches the primary page. **corrected** = the entry needs a change.
**new** = a suggested addition, verified to exist with exactly the metadata given.

## A. New citations (suggested additions)

| key | full reference | verified URL | status | Shared | Differs |
|---|---|---|---|---|---|
| woodworth2020kernel | Blake Woodworth, Suriya Gunasekar, Jason D. Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Daniel Soudry, Nathan Srebro. Kernel and Rich Regimes in Overparametrized Models. COLT 2020, PMLR 125:3635–3673, 2020. | https://proceedings.mlr.press/v125/woodworth20a.html | new | A single scale parameter decides between a kernel regime and a rich regime with different implicit biases. Our small-scale and large-scale preferences are the same kind of scale-controlled dichotomy. | Their scale is the initialisation scale, fixed for the whole run, in homogeneous (mainly deep linear) models. Ours is the output scale, which grows during training in a non-homogeneous width-one sine unit, and the switch is between two minimisers of the fixed-scale loss. |
| geiger2020disentangling | Mario Geiger, Stefano Spigler, Arthur Jacot, Matthieu Wyart. Disentangling feature and lazy training in deep neural networks. Journal of Statistical Mechanics: Theory and Experiment 2020(11):113301, 2020. doi:10.1088/1742-5468/abc4de | https://doi.org/10.1088/1742-5468/abc4de (Crossref record: https://api.crossref.org/works/10.1088/1742-5468/abc4de); arXiv 1906.08034 | new | The scale of the network output (a last-layer multiplier) controls the crossover between lazy and feature learning. This is the same knob that indexes our fixed-scale objective. | They map an empirical crossover that depends on width, across wide architectures and real datasets. We certify one switch scale in a width-one network and predict when output-scale growth within a run crosses it. |
| saxe2014exact | Andrew M. Saxe, James L. McClelland, Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. ICLR 2014 (conference track). arXiv:1312.6120. | https://iclr.cc/archive/2014/conference-proceedings/ ; https://arxiv.org/abs/1312.6120 | new | Exact analysis of a minimal model in which training shows long plateaus followed by sudden transitions. | Their transitions occur in time, from small initialisation, in deep linear networks. Ours is a change of the preferred hidden feature as output scale increases in a non-linear, non-monotone unit, and its timing is set by a tracking lag rather than by escape from a plateau. |
| abbe2023sgd | Emmanuel Abbe, Enric Boix Adserà, Theodor Misiakiewicz. SGD learning on neural networks: leap complexity and saddle-to-saddle dynamics. COLT 2023, PMLR 195:2552–2623, 2023. | https://proceedings.mlr.press/v195/abbe23a.html | new | Features are acquired in discrete stages with saddle-to-saddle dynamics, so the "staged learning" family is not limited to linear models. | Their stage times scale with input dimension through the target's leap complexity on isotropic high-dimensional data. Our single transition is driven by output-scale growth on a one-dimensional input, and its location is a property of the fixed-scale loss. |
| krupa2001extending | M. Krupa, P. Szmolyan. Extending Geometric Singular Perturbation Theory to Nonhyperbolic Points—Fold and Canard Points in Two Dimensions. SIAM Journal on Mathematical Analysis 33(2):286–314, 2001. doi:10.1137/S0036141099360919 | https://doi.org/10.1137/S0036141099360919 (Crossref record: https://api.crossref.org/works/10.1137/S0036141099360919) | new | Slow–fast theory for a tracked branch carried by a slowly varied parameter to a fold, where Fenichel's normally hyperbolic theory (which we cite for the first-order lag) stops applying. A reviewer asking about bifurcation delay would expect this reference. | They treat planar continuous-time normal forms with an explicit small parameter. Our slow variable is the trained output scale itself, discrete Adam/SGD steps are outside their theorems, and the submitted lag law is only the first-order, off-fold correction. |
| johansson2017arb | Fredrik Johansson. Arb: Efficient Arbitrary-Precision Midpoint-Radius Interval Arithmetic. IEEE Transactions on Computers 66(8):1281–1292, 2017. doi:10.1109/TC.2017.2690633 | https://doi.org/10.1109/TC.2017.2690633 (Crossref record: https://api.crossref.org/works/10.1109/TC.2017.2690633); arXiv 1611.02831 | new | Our independent checker does every decision as an Arb ball-arithmetic comparison through python-flint, and the paper names Arb without citing it. | Arb provides rigorous enclosures of arithmetic only. The certificate chain (profiled loss, gap enclosures, Krawczyk boxes, ring and positive-definite checks) and its mathematical reductions are ours, and Arb does not validate them. |
| krawczyk1969newton | R. Krawczyk. Newton-Algorithmen zur Bestimmung von Nullstellen mit Fehlerschranken. Computing 4(3):187–201, 1969. doi:10.1007/BF02234767 | https://doi.org/10.1007/BF02234767 (Crossref record: https://api.crossref.org/works/10.1007/BF02234767) | new | Source of the interval Newton (Krawczyk) operator behind the "first-order Krawczyk boxes" that our checker verifies. The paper uses the term without a citation. | A general method for enclosing zeros of a map. Which map is certified (the stationarity conditions at the switch) and how the boxes feed the switch bracket are specific to our chain. |
| bertinetto2021preface | Luca Bertinetto, João F. Henriques, Samuel Albanie, Michela Paganini, Gül Varol. Preface. NeurIPS 2020 Workshop on Pre-registration in Machine Learning, PMLR 148:i–i, 2021. | https://proceedings.mlr.press/v148/bertinetto21a.html | new | Pre-registration in ML means committing hypotheses and protocols before experiments, which is the practice our registered tests follow. | That workshop used a peer-reviewed, two-stage protocol. Our registrations are self-registered repository commits with timestamps, not an external registry, as §8 already states. |

## B. Citations in the submitted related work (§7 and Appendix T)

| key | full reference (as in the submitted PDF) | verified URL | status | Shared | Differs |
|---|---|---|---|---|---|
| rosset2003margin | Saharon Rosset, Ji Zhu, Trevor Hastie. Margin maximizing loss functions. NeurIPS 16, MIT Press, 2003. | https://proceedings.neurips.cc/paper/2003/hash/0fe473396242072e84af286632d3f0ff-Abstract.html | verified | Regularised logistic minimisers approach the max-margin separator as regularisation vanishes. This is the large-scale end of our fixed-scale path. | Theirs is a limit statement. We locate a finite-scale switch between placed and unplaced minimisers and the training lag behind it. |
| ji2020gradient | Ziwei Ji, Miroslav Dudík, Robert E. Schapire, Matus Telgarsky. Gradient descent follows the regularization path for general losses. COLT 2020, PMLR 125:2109–2136, 2020. | https://proceedings.mlr.press/v125/ji20a.html | verified | Gradient descent and the regularisation path share a limit direction, which justifies comparing training with fixed-scale minimisers. | Their result covers linear predictors and limit directions. Ours is a nonlinear unit with a finite-scale switch and a measured lag. |
| atanasov2025optimization | Alexander Atanasov, Alexandru Meterez, James Simon, Cengiz Pehlevan. The optimization landscape of SGD across the feature learning strength. ICLR 2025, pp. 5740–5779. | https://proceedings.iclr.cc/paper_files/paper/2025/hash/112c37467b801ad82d5e9c2ea4ff73f2-Abstract-Conference.html | verified | Output down-scaling sets feature-learning strength and changes SGD's behaviour. | They fix the scale per run and study standard architectures empirically. Our scale grows within a run, and the switch it crosses is certified. |
| moroshko2020implicit | Edward Moroshko, Blake E Woodworth, Suriya Gunasekar, Jason D Lee, Nati Srebro, Daniel Soudry. Implicit bias in deep linear classification: Initialization scale vs training accuracy. NeurIPS 33, pp. 22182–22193, 2020. | https://proceedings.neurips.cc/paper/2020/hash/fc2022c89b61c76bbef978f1370660bf-Abstract.html | verified | In classification, a scale variable decides between kernel-like and rich implicit bias. | Their models are deep linear (homogeneous) classifiers and they characterise the limit bias. Ours is a non-monotone nonlinear width-one unit with a switch at finite scale. |
| chizat2019on | Lénaïc Chizat, Edouard Oyallon, Francis Bach. On lazy training in differentiable programming. NeurIPS 32, 2019. | https://proceedings.neurips.cc/paper_files/paper/2019/hash/ae614c557843b1df326cb29c57225459-Abstract.html | verified | Output scaling organises qualitatively different training regimes. | Lazy training is linearisation around initialisation under large output scaling. Both of our regimes are defined by minimisers of the fixed-scale loss, and the hidden unit moves substantially at the switch. |
| kunin2024getrich | Daniel Kunin, Allan Raventós, Clémentine Dominé, Feng Chen, David Klindt, Andrew Saxe, Surya Ganguli. Get rich quick: exact solutions reveal how unbalanced initializations promote rapid feature learning. NeurIPS 37, pp. 81157–81203, 2024. doi:10.52202/079017-2580 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/94074dd5a072d28ff75a76dabed43767-Abstract-Conference.html | verified | Exact solutions in minimal models show that relative layer scales set how fast features are learned. | Their control is initialisation imbalance. Ours is output-scale growth relative to hidden relaxation (χ), measured during training. |
| kumar2024grokking | Tanishq Kumar, Blake Bordelon, Samuel Gershman, Cengiz Pehlevan. Grokking as the transition from lazy to rich training dynamics. ICLR 2024, pp. 23010–23035. | https://proceedings.iclr.cc/paper_files/paper/2024/hash/63ed15a46a143ff57484b38cd6b85d91-Abstract-Conference.html | verified | The scale of the network output governs when features are learned. | They explain delayed generalisation (grokking) along the training trajectory. We locate a switch of the fixed-scale training-loss minimiser and predict the lag behind it, and we make no generalisation claim. |
| lyu2024dichotomy | Kaifeng Lyu, Jikai Jin, Zhiyuan Li, Simon Du, Jason Lee, Wei Hu. Dichotomy of early and late phase implicit biases can provably induce grokking. ICLR 2024, pp. 33897–33936. | https://proceedings.iclr.cc/paper_files/paper/2024/hash/909c8fef63e1cede406ce9e6794f99a2-Abstract-Conference.html | verified | An early kernel-like bias gives way to a late margin bias. This is the same small-to-large progression as our class-mean to worst-case-margin preference. | They prove the transition time for grokking in specific settings. We certify where the fixed-scale preference switches and measure and predict how far training lags behind it. |
| kunin2025alternating | Daniel Kunin, Giovanni Luca Marchetti, Feng Chen, Dhruva Karkada, James Simon, Michael Deweese, Surya Ganguli, Nina Miolane. Alternating gradient flows: A theory of feature learning in two-layer neural networks. NeurIPS 38, pp. 4377–4424, 2025. doi:10.52202/085713-0156 | https://proceedings.neurips.cc/paper_files/paper/2025/hash/06cbd2e81dfbd3bb4cb0abce95b32584-Abstract-Conference.html | verified | A theory of when individual hidden units change role and of the resulting loss drops. | AGF describes dormant units activating from small initialisation. Our single unit is always active, and what changes is which feature it implements, driven by output scale. |
| morwani2024feature | Depen Morwani, Benjamin Edelman, Costin-Andrei Oncescu, Rosie Zhao, Sham Kakade. Feature emergence via margin maximization: case studies in algebraic tasks. ICLR 2024, pp. 29077–29114. | https://proceedings.iclr.cc/paper_files/paper/2024/hash/7d4647db960780c8b5a4e7d9e4a58d68-Abstract-Conference.html | verified | Margin maximisation determines which features are learned. | They characterise features at the max-margin limit. Our switch happens at finite scale, before the margin limit. |
| lyu2021gradient | Kaifeng Lyu, Zhiyuan Li, Runzhe Wang, Sanjeev Arora. Gradient descent on two-layer nets: Margin maximization and simplicity bias. NeurIPS 34, pp. 12978–12991, 2021. | https://proceedings.neurips.cc/paper/2021/hash/6c351da15b5e8a743a21ee96a86e25df-Abstract.html | verified | An early simple solution precedes margin maximisation in two-layer networks. | They study leaky-ReLU networks on linearly separable data. Our task cannot be solved by any monotone activation, and the switch scale between the two preferences is computed. |
| shah2020pitfalls | Harshay Shah, Kaustav Tamuly, Aditi Raghunathan, Prateek Jain, Praneeth Netrapalli. The pitfalls of simplicity bias in neural networks. NeurIPS 33, pp. 9573–9585, 2020. | https://proceedings.neurips.cc/paper/2020/hash/6cfe0e6127fa25df2a0ef2ae1067d915-Abstract.html | verified | Simplicity bias selects a simpler feature over a more robust one. Our exploratory extension uses their linear-plus-slab benchmark. | They document the bias. We predict a fixed-scale switch toward the slab feature, and our registered transfer test on their benchmark was unresolved. |
| tsoy2024simplicity | Nikita Tsoy, Nikola Konstantinov. Simplicity bias of two-layer networks beyond linearly separable data. ICML 2024, PMLR 235:48728–48767. | https://proceedings.mlr.press/v235/tsoy24a.html | verified | Simplicity bias on data that are not linearly separable. Our task is also not solvable by a monotone unit. | They analyse the early phase from small initialisation. Our switch is later, driven by output-scale growth. |
| atanasov2022neural | Alexander Atanasov, Blake Bordelon, Cengiz Pehlevan. Neural networks as kernel learners: The silent alignment effect. ICLR 2022. | https://iclr.cc/virtual/2022/poster/7005 | verified | Early alignment of features is separate from later growth in scale. | In silent alignment the kernel aligns early and then only grows. In our setting, growth in scale itself changes the preferred feature. |
| kalimeris2019sgd | Dimitris Kalimeris, Gal Kaplun, Preetum Nakkiran, Benjamin Edelman, Tristan Yang, Boaz Barak, Haofeng Zhang. SGD on neural networks learns functions of increasing complexity. NeurIPS 32, 2019. | https://proceedings.neurips.cc/paper/2019/hash/b432f34c5a997c8e7c806a895ecc5e25-Abstract.html | verified | An early simple (linear) classifier is learned first and retained. | Their evidence is empirical, measured with information-theoretic metrics on standard networks. We prove the small-scale preference and certify the switch away from it. |
| refinetti2023neural | Maria Refinetti, Alessandro Ingrosso, Sebastian Goldt. Neural networks trained with SGD learn distributions of increasing complexity. ICML 2023, PMLR 202:28843–28863. | https://proceedings.mlr.press/v202/refinetti23a.html | verified | Lower-order statistics are used first. This is the closest prior result to our small-scale class-mean preference (Theorem 1). | Their ordering is in training time and over input statistics. Ours is in output scale, for a fixed finite objective, and is proved. |
| jacot2021saddle | Arthur Jacot, François Ged, Berfin Şimşek, Clément Hongler, Franck Gabriel. Saddle-to-saddle dynamics in deep linear networks: Small initialization training, symmetry, and sparsity. arXiv:2106.15933, 2021. | https://arxiv.org/abs/2106.15933 | verified | Staged transitions in a minimal model. | Their transitions are in time, through rank-increasing saddles of deep linear networks. Ours is a single feature change in a non-linear unit, organised by output scale. |
| pesme2023saddle | Scott Pesme, Nicolas Flammarion. Saddle-to-saddle dynamics in diagonal linear networks. NeurIPS 36, pp. 7475–7505, 2023. doi:10.52202/075280-0329 | https://proceedings.neurips.cc/paper_files/paper/2023/hash/17a9ab4190289f0e1504bbb98d1d111a-Abstract-Conference.html | verified | An exact account of discrete transitions in a small model. | Their saddles and jump times are derived exactly along the flow. Our transition is in output scale on the conditional loss, and training's relation to it is predicted to first order and then tested. |
| jin2023understanding | Jikai Jin, Zhiyuan Li, Kaifeng Lyu, Simon Shaolei Du, Jason D. Lee. Understanding incremental learning of gradient descent: A fine-grained analysis of matrix sensing. ICML 2023, PMLR 202:15200–15238. | https://proceedings.mlr.press/v202/jin23a.html | verified | Staged (incremental) learning under gradient descent. | Matrix sensing from small initialisation, with rank-by-rank increments. Ours is a placement switch of one hidden unit. |
| glasgow2024sgd | Margalit Glasgow. SGD finds then tunes features in two-layer neural networks with near-optimal sample complexity: A case study in the XOR problem. ICLR 2024, pp. 52419–52430. | https://proceedings.iclr.cc/paper_files/paper/2024/hash/e6d37cc5723e810b793c834bcb6647cf-Abstract-Conference.html | verified | Features are acquired in phases in a minimal task that monotone linear features cannot solve. | Her focus is the sample complexity of SGD on XOR. Ours is where the fixed-scale preference switches and the lag of training behind it. |
| fenichel1979geometric | Neil Fenichel. Geometric singular perturbation theory for ordinary differential equations. Journal of Differential Equations 31(1):53–98, 1979. doi:10.1016/0022-0396(79)90152-9 | https://doi.org/10.1016/0022-0396(79)90152-9 (Crossref record: https://api.crossref.org/works/10.1016/0022-0396(79)90152-9) | verified | The source of the first-order slow-manifold correction in Eq. (6). | It is continuous-time and normally hyperbolic. It gives no guarantee for discrete Adam/SGD, which is why we test the first-order law prospectively. |
| kunin2023asymmetric | Daniel Kunin, Atsushi Yamamura, Chao Ma, Surya Ganguli. The asymmetric maximum margin bias of quasi-homogeneous neural networks. ICLR 2023. | https://iclr.cc/virtual/2023/poster/10967 | verified | Our network is quasi-homogeneous in its output layer, which is the model class they analyse. | Their margin results concern the separable limit. Our switch happens at finite scale, before the relevant branch separates. |
| cai2025implicit | Yuhang Cai, Kangjie Zhou, Jingfeng Wu, Song Mei, Michael Lindsey, Peter Bartlett. Implicit bias of gradient descent for non-homogeneous deep networks. ICML 2025, PMLR 267:6426–6504. | https://proceedings.mlr.press/v267/cai25m.html | verified | Margin behaviour of non-homogeneous networks, a class that includes our unit. | Their results begin once the empirical risk is small (the separable regime). Our switch happens before that regime. |
| chiang2023loss | Ping-yeh Chiang, Renkun Ni, David Y. Miller, Arpit Bansal, Jonas Geiping, Micah Goldblum, Tom Goldstein. Loss landscapes are all you need: Neural network generalization can be explained without the implicit bias of gradient descent. ICLR 2023. | https://iclr.cc/virtual/2023/oral/12746 | verified | An outcome is attributed to loss geometry rather than to the optimiser. | They explain generalisation statically. We add a dynamical lag that depends on the optimiser (P, η) on top of the static switch. |
| garipov2018loss | Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry Vetrov, Andrew G. Wilson. Loss surfaces, mode connectivity, and fast ensembling of DNNs. NeurIPS 31, 2018. | https://proceedings.neurips.cc/paper_files/paper/2018/hash/be3087e74e9100d4bc4c6268cdbe8456-Abstract.html | verified | Low-loss connecting paths between minima. | They show such paths exist. Ours is a path of minimisers indexed by scale, together with how training follows it. |
| draxler2018essentially | Felix Draxler, Kambis Veschgini, Manfred Salmhofer, Fred Hamprecht. Essentially no barriers in neural network energy landscape. ICML 2018, PMLR 80:1309–1318. | https://proceedings.mlr.press/v80/draxler18a.html | verified | Paths between minima in the loss landscape. | They study the existence of low-barrier paths. We study dynamic tracking along a scale-indexed branch. |
| schaeffer2023are | Rylan Schaeffer, Brando Miranda, Sanmi Koyejo. Are emergent abilities of large language models a mirage? NeurIPS 36, pp. 55565–55581, 2023. doi:10.52202/075280-2425 | https://proceedings.neurips.cc/paper_files/paper/2023/hash/adc98a266f45005c403b8311ca7e8bd7-Abstract-Conference.html | verified | A binary metric can make a gradual change look like a sharp onset. Our continuous-error check addresses exactly this. | They study LLM benchmarks. In our runs, 76% of the error improvement happens while no run solves, so we withdraw any sharp capability-onset reading. |
| naitzat2020topology | Gregory Naitzat, Andrey Zhitnikov, Lek-Heng Lim. Topology of deep neural networks. JMLR 21(184):1–40, 2020. | https://jmlr.org/papers/v21/20-345.html | verified | Narrow networks change the topology of class arrangements, depending on the activation. | They track Betti numbers empirically in trained networks. Our task has an exact one-dimensional obstruction (Proposition 1), and the training question is placement. |
| guss2018on | William H. Guss, Ruslan Salakhutdinov. On characterizing the capacity of neural networks using algebraic topology. arXiv:1802.04443, 2018. | https://arxiv.org/abs/1802.04443 | verified | Sharp transitions in what an architecture achieves on data with a known topology. | Their transitions are in architecture width against data complexity. Ours is in output scale at a fixed architecture. |
| ren2026low | Junyu Ren, Lek-Heng Lim. Low-dimensional topology of deep neural networks. Proceedings of the 43rd ICML, **PMLR 306:104476–104499**, 2026. | https://proceedings.mlr.press/v306/ren26g.html ; https://arxiv.org/abs/2606.31856 | **corrected** (volume and pages added; the PDF had neither because the volume was not yet online) | Monotone activations are obstructed in narrow networks and folding activations are not. Proposition 1 is the one-dimensional instance of this. | Their theorems concern representability. We study which fixed-scale minimiser training reaches and when. |
| johnson2019deep | Jesse Johnson. Deep, skinny neural networks are not universal approximators. ICLR 2019. | https://iclr.cc/Conferences/2019/Schedule?showEvent=905 | verified | Narrow width imposes topological obstructions. | Approximation existence, not training. |
| hanin2017approximating | Boris Hanin, Mark Sellke. Approximating continuous functions by ReLU nets of minimal width. arXiv:1710.11278, 2017. | https://arxiv.org/abs/1710.11278 | verified | Width bounds for representability. | They bound the minimal width for approximation. We take representability as given and study training. |
| park2021minimum | Sejun Park, Chulhee Yun, Jaeho Lee, Jinwoo Shin. Minimum width for universal approximation. ICLR 2021. | https://iclr.cc/virtual/2021/poster/2920 | verified | Minimum-width obstructions. | Approximation theory, with no dynamics. |
| cai2023achieve | Yongqiang Cai. Achieve the minimum width of neural networks for universal approximation. ICLR 2023. | https://iclr.cc/virtual/2023/poster/12225 | verified | Minimum-width obstructions. | Approximation theory, with no dynamics. |
| kim2024minimum | Namjun Kim, Chanho Min, Sejun Park. Minimum width for universal approximation using ReLU networks on compact domain. ICLR 2024. | https://proceedings.iclr.cc/paper_files/paper/2024/hash/0d61c5f5ef91e7e8a091b7b8f72b853c-Abstract-Conference.html | verified | Minimum-width obstructions. | Approximation theory for ReLU, with no dynamics. |
| kidger2020universal | Patrick Kidger, Terry Lyons. Universal approximation with deep narrow networks. COLT 2020, PMLR 125:2306–2327. | https://proceedings.mlr.press/v125/kidger20a.html | verified | Narrow networks and their limits. | Approximation existence, not training. |
| li2023minimum | Li'Ang Li, Yifei Duan, Guanghua Ji, Yongqiang Cai. Minimum width of Leaky-ReLU neural networks for uniform universal approximation. ICML 2023, PMLR 202:19460–19470. | https://proceedings.mlr.press/v202/li23g.html | verified | Minimum-width obstructions. | Approximation theory, with no dynamics. |
| chen2018neural | Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David K. Duvenaud. Neural ordinary differential equations. NeurIPS 31, 2018. | https://proceedings.neurips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html | verified | Continuous-depth flows inherit dimension-induced topological restrictions. | They introduce a model class. Our obstruction is in a discrete width-one map. |
| dupont2019augmented | Emilien Dupont, Arnaud Doucet, Yee Whye Teh. Augmented neural ODEs. NeurIPS 32, 2019. | https://proceedings.neurips.cc/paper_files/paper/2019/hash/21be9a4bd4f81549a9d1d241981cec3c-Abstract.html | verified | A restriction that state augmentation removes. Our non-monotone activation plays the corresponding role. | They remove the obstruction by architecture. We study how training realises the solution once it is representable. |
| telgarsky2016benefits | Matus Telgarsky. Benefits of depth in neural networks. COLT 2016, PMLR 49:1517–1539. | https://proceedings.mlr.press/v49/telgarsky16.html | verified | Oscillation and folding as the source of expressivity. | Depth separation measures representation, not trained outcomes. |
| eldan2016the | Ronen Eldan, Ohad Shamir. The power of depth for feedforward neural networks. COLT 2016, PMLR 49:907–940. | https://proceedings.mlr.press/v49/eldan16.html | verified | Expressivity gaps between architectures. | Representation only. |
| montufar2014on | Guido Montúfar, Razvan Pascanu, Kyunghyun Cho, Yoshua Bengio. On the number of linear regions of deep neural networks. NeurIPS 27, 2014. | https://proceedings.neurips.cc/paper_files/paper/2014/hash/fa6f2a469cc4d61a92d96e74617c3d2a-Abstract.html | verified | Folding increases the complexity a network can represent. | They count linear regions. We study which region structure training reaches. |
| raghu2017on | Maithra Raghu, Ben Poole, Jon Kleinberg, Surya Ganguli, Jascha Sohl-Dickstein. On the expressive power of deep neural networks. ICML 2017, PMLR 70:2847–2854. | https://proceedings.mlr.press/v70/raghu17a.html | verified | Expressivity measures for trained networks. | Representation complexity, not a training threshold. |
| shalevshwartz2017failures | Shai Shalev-Shwartz, Ohad Shamir, Shaked Shammah. Failures of gradient-based deep learning. ICML 2017, PMLR 70:3067–3075. | https://proceedings.mlr.press/v70/shalev-shwartz17a.html | verified | Expressible targets that gradient methods fail to learn. | Their failures come from weak gradient signal. Ours is a scale-dependent preference for the unplaced feature, with an informative gradient throughout. |
| hendrycks2016gaussian | Dan Hendrycks, Kevin Gimpel. Gaussian error linear units (GELUs). arXiv:1606.08415, 2016. | https://arxiv.org/abs/1606.08415 | verified | A non-monotone activation. We validate a conditional switch for GELU. | GELU was motivated by performance. Our GELU training tests are unresolved or rejected. |
| ramachandran2017searching | Prajit Ramachandran, Barret Zoph, Quoc V. Le. Searching for activation functions. arXiv:1710.05941, 2017. | https://arxiv.org/abs/1710.05941 | verified | A non-monotone activation (SiLU/Swish), included in our switch validation. | It was found by architecture search. In our tests, SiLU grows too fast for the tracking regime. |
| misra2020mish | Diganta Misra. Mish: A self regularized non-monotonic activation function. BMVC 2020. doi:10.5244/C.34.191 | https://bmva-archive.org.uk/bmvc/2020/conference/papers/paper_0928.html | verified | A non-monotone activation, included in our switch validation. | It was motivated by performance. In our tests, Mish also grows too fast for the tracking regime. |
| sitzmann2020implicit | Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, Gordon Wetzstein. Implicit neural representations with periodic activation functions. NeurIPS 33, pp. 7462–7473, 2020. | https://proceedings.neurips.cc/paper_files/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html | verified | A sine activation. | They use it for signal representation. Our tunable dip a in t + a·sin t yields an analytic boundary and a scaling reduction. |
| ziyin2020neural | Liu Ziyin, Tilman Hartwig, Masahito Ueda. Neural networks fail to learn periodic functions and how to fix it. NeurIPS 33, pp. 1583–1594, 2020. | https://proceedings.neurips.cc/paper_files/paper/2020/hash/1160453108d3e537255e9f7b931f4e90-Abstract.html | verified | A linear-plus-periodic activation (Snake), close in form to our f_a. | Snake was motivated by extrapolating periodic functions. We use the dip parameter as a control variable for the training threshold. |
| kaplan2020scaling | Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei. Scaling laws for neural language models. arXiv:2001.08361, 2020. | https://arxiv.org/abs/2001.08361 | verified | Power-law relations between training resources and outcomes. | Their laws concern compute and loss. Our placement onset depends on budget, and its effective exponent does not extrapolate. |
| zhu2023minimalist | Xingyu Zhu, Zixuan Wang, Xiang Wang, Mo Zhou, Rong Ge. Understanding edge-of-stability training dynamics with a minimalist example. ICLR 2023. | https://iclr.cc/virtual/2023/poster/11908 | verified | A minimal model that makes training dynamics analysable. | Their subject is EoS. Our constructed-solution measurements separate EoS from the observed delay. |
| cohen2021gradient | Jeremy Cohen, Simran Kaur, Yuanzhi Li, Zico Kolter, Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. ICLR 2021. | https://iclr.cc/virtual/2021/poster/2577 | verified | A dynamical effect of the optimiser that is absent from the static loss. | EoS is a sharpness phenomenon under full-batch GD. Our lag is a slow-manifold tracking effect, measured under Adam and SGD. |
| ahn2022understanding | Kwangjun Ahn, Jingzhao Zhang, Suvrit Sra. Understanding the unstable convergence of gradient descent. ICML 2022, PMLR 162:247–257. | https://proceedings.mlr.press/v162/ahn22a.html | verified | Which stationary points GD can converge to depends on step size. | Their result is about non-convergence to sharp points. Our tracked branch has a positive-definite Hessian. |
| ahn2023threshold | Kwangjun Ahn, Sebastien Bubeck, Sinho Chewi, Yin Tat Lee, Felipe Suarez, Yi Zhang. Learning threshold neurons via edge of stability. NeurIPS 36, pp. 19540–19569, 2023. doi:10.52202/075280-0858 | https://proceedings.neurips.cc/paper_files/paper/2023/hash/3e592c571de69a43d7a870ea89c7e33a-Abstract-Conference.html | verified | A sharp transition decides whether a unit acquires a useful bias (placement). | Their control parameter is the step size, through EoS. Ours is output scale, through the fixed-scale loss. |
| rubin2024grokking | Noa Rubin, Inbar Seroussi, Zohar Ringel. Grokking as a first order phase transition in two layer networks. ICLR 2024, pp. 23881–23904. | https://proceedings.iclr.cc/paper_files/paper/2024/hash/682f87a8c306098ec8be29019bd76aa4-Abstract-Conference.html | verified | A first-order-type switch between feature solutions. | Theirs is a teacher–student phase transition in a large-width limit. Ours is a certified switch at finite width one. |
| liu2023omnigrok | Ziming Liu, Eric Michaud, Max Tegmark. Omnigrok: Grokking beyond algorithmic data. ICLR 2023. | https://iclr.cc/virtual/2023/oral/12716 | verified | A norm/scale variable organises a qualitative change. | Their variable indexes a train–test mismatch. Ours indexes which training-loss minimiser is placed. |
| prieto2025grokking | Lucas Prieto, Melih Barsbey, Pedro Mediano, Tolga Birdal. Grokking at the edge of numerical stability. ICLR 2025, pp. 81151–81168. | https://proceedings.iclr.cc/paper_files/paper/2025/hash/c9e6ac15e689e06139d7b39e1667b165-Abstract-Conference.html | verified | Logit-scale growth as the variable behind a delayed transition. | In their account the growth delays generalisation and ends in numerical collapse. In ours, growth makes placement preferred. |
| nanda2023progress | Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. ICLR 2023. | https://iclr.cc/virtual/2023/poster/11385 | verified | Progress that is hidden before an apparent transition. Our output scale grows before placement. | Their measures are mechanistic, on modular addition. Ours is an analytic margin decomposition. |
| barak2022hidden | Boaz Barak, Benjamin Edelman, Surbhi Goel, Sham Kakade, Eran Malach, Cyril Zhang. Hidden progress in deep learning: SGD learns parities near the computational limit. NeurIPS 35, pp. 21750–21764, 2022. doi:10.52202/068431-1581 | https://proceedings.neurips.cc/paper_files/paper/2022/hash/884baf65392170763b27c914087bde01-Abstract-Conference.html | verified | An abrupt change preceded by steady hidden progress. | Their delay is computational (parity). Ours is a scale threshold of the loss plus a tracking lag. |
| soudry2018the | Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, Nathan Srebro. The implicit bias of gradient descent on separable data. JMLR 19(70):1–57, 2018. | https://jmlr.org/papers/v19/18-188.html | verified | Margin growth with norm on separable data. This is the large-scale context. | Their predictors are linear and homogeneous, at the separable limit. Our unit is non-homogeneous and switches at finite scale. |
| lyu2020gradient | Kaifeng Lyu, Jian Li. Gradient descent maximizes the margin of homogeneous neural networks. ICLR 2020. | https://iclr.cc/virtual_2020/poster_SJeLIgBKPS.html | verified | Normalised margin increases under gradient descent at large scale. | Their networks are homogeneous, and the result holds after separation. Our switch comes before separation. |

## C. BibTeX (corrected and new entries only)

```bibtex
% corrected: volume and pages now online. Verified: https://proceedings.mlr.press/v306/ren26g.html (2026-11-14)
@inproceedings{ren2026low,
  author        = {Junyu Ren and Lek-Heng Lim},
  title         = {Low-dimensional topology of deep neural networks},
  booktitle     = {Proceedings of the 43rd International Conference on Machine Learning},
  series        = {Proceedings of Machine Learning Research},
  volume        = {306},
  pages         = {104476--104499},
  publisher     = {PMLR},
  year          = {2026},
  eprint        = {2606.31856},
  archivePrefix = {arXiv}
}

% verified: https://proceedings.mlr.press/v125/woodworth20a.html (2026-11-14)
@inproceedings{woodworth2020kernel,
  author    = {Blake Woodworth and Suriya Gunasekar and Jason D. Lee and Edward Moroshko and Pedro Savarese and Itay Golan and Daniel Soudry and Nathan Srebro},
  title     = {Kernel and Rich Regimes in Overparametrized Models},
  booktitle = {Proceedings of Thirty Third Conference on Learning Theory},
  series    = {Proceedings of Machine Learning Research},
  volume    = {125},
  pages     = {3635--3673},
  publisher = {PMLR},
  year      = {2020}
}

% verified: Crossref record for doi:10.1088/1742-5468/abc4de (2026-11-14); arXiv 1906.08034
@article{geiger2020disentangling,
  author  = {Mario Geiger and Stefano Spigler and Arthur Jacot and Matthieu Wyart},
  title   = {Disentangling feature and lazy training in deep neural networks},
  journal = {Journal of Statistical Mechanics: Theory and Experiment},
  volume  = {2020},
  number  = {11},
  pages   = {113301},
  doi     = {10.1088/1742-5468/abc4de},
  year    = {2020}
}

% verified: https://iclr.cc/archive/2014/conference-proceedings/ (conference track) and https://arxiv.org/abs/1312.6120 (2026-11-14)
@inproceedings{saxe2014exact,
  author        = {Andrew M. Saxe and James L. McClelland and Surya Ganguli},
  title         = {Exact solutions to the nonlinear dynamics of learning in deep linear neural networks},
  booktitle     = {International Conference on Learning Representations (ICLR)},
  year          = {2014},
  eprint        = {1312.6120},
  archivePrefix = {arXiv}
}

% verified: https://proceedings.mlr.press/v195/abbe23a.html (2026-11-14)
@inproceedings{abbe2023sgd,
  author    = {Emmanuel Abbe and Enric {Boix Adser{\`a}} and Theodor Misiakiewicz},
  title     = {{SGD} learning on neural networks: leap complexity and saddle-to-saddle dynamics},
  booktitle = {Proceedings of Thirty Sixth Conference on Learning Theory},
  series    = {Proceedings of Machine Learning Research},
  volume    = {195},
  pages     = {2552--2623},
  publisher = {PMLR},
  year      = {2023}
}

% verified: Crossref record for doi:10.1137/S0036141099360919 (2026-11-14)
@article{krupa2001extending,
  author  = {M. Krupa and P. Szmolyan},
  title   = {Extending Geometric Singular Perturbation Theory to Nonhyperbolic Points---Fold and Canard Points in Two Dimensions},
  journal = {SIAM Journal on Mathematical Analysis},
  volume  = {33},
  number  = {2},
  pages   = {286--314},
  doi     = {10.1137/S0036141099360919},
  year    = {2001}
}

% verified: Crossref record for doi:10.1109/TC.2017.2690633 (2026-11-14); arXiv 1611.02831
@article{johansson2017arb,
  author  = {Fredrik Johansson},
  title   = {Arb: Efficient Arbitrary-Precision Midpoint-Radius Interval Arithmetic},
  journal = {IEEE Transactions on Computers},
  volume  = {66},
  number  = {8},
  pages   = {1281--1292},
  doi     = {10.1109/TC.2017.2690633},
  year    = {2017}
}

% verified: Crossref record for doi:10.1007/BF02234767 (2026-11-14)
@article{krawczyk1969newton,
  author  = {R. Krawczyk},
  title   = {{Newton-Algorithmen zur Bestimmung von Nullstellen mit Fehlerschranken}},
  journal = {Computing},
  volume  = {4},
  number  = {3},
  pages   = {187--201},
  doi     = {10.1007/BF02234767},
  year    = {1969}
}

% verified: https://proceedings.mlr.press/v148/bertinetto21a.html (2026-11-14)
@inproceedings{bertinetto2021preface,
  author    = {Luca Bertinetto and Jo{\~a}o F. Henriques and Samuel Albanie and Michela Paganini and G{\"u}l Varol},
  title     = {Preface},
  booktitle = {NeurIPS 2020 Workshop on Pre-registration in Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {148},
  pages     = {i--i},
  publisher = {PMLR},
  year      = {2021}
}
```

## D. Caveats for the writer

- `krupa2001extending`: I checked only the metadata and the topic (extending slow manifolds past fold and canard
  points by blow-up). Before citing a specific theorem or constant (for example an ε^{2/3} or Airy-type passage
  statement), read it in the paper itself.
- `krawczyk1969newton`: cite it only as the origin of the operator. I did not verify who first established its use
  as an existence-and-uniqueness test, so do not attribute that to it.
- `saxe2014exact`: the ICLR 2014 conference-proceedings page lists it under Conference Proceedings. There are no
  pages or volume.
- `bertinetto2021preface`: this is the workshop's editorial preface (PMLR 148:i), the only citable item for the
  workshop as a whole. The PMLR page has no abstract. The two-stage format (papers accepted on an experimental
  protocol that is reviewed before results exist) is confirmed by the workshop page,
  https://neurips.cc/virtual/2020/workshop/16158 ("The pre-registration experiment: an alternative publication
  model for machine learning research"; same five organisers).
- No DBLP or OpenReview page was used (DBLP returned a bot challenge). ICLR 2022/2023 entries are verified on
  iclr.cc, as in the earlier verification.
