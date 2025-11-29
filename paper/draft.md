 # Integer participation, structural noise, and discrete topology in superconducting and superfluid clusters

## Abstract

We study how structural noise and topology shape the “Mother Frequency”  
\(F_m = \Theta_D / T_c\) in a heterogeneous set of superconductors and superfluid helium.  
Building on a previously proposed delayed-oscillator locking framework, we introduce an
integer participation model in which each material contributes an effective number of
coherent “jumps” \(N\) to a common oscillatory backbone. A robust base frequency
\(f_{\text{base}}\) is calibrated per family by minimizing the median distance of  
\(N = F_m^* / f_{\text{base}}\) to the nearest integer, where \(F_m^*\) is corrected for
family-dependent structural noise estimated from an independent simulator.
We then ask a purely empirical question: are the corrected participations significantly
more integer and less noisy than expected from simple null models?

Across \(\sim 250\) superconducting and superfluid entries, the distribution of  
\(|\delta| = |N - \text{round}(N)|\) is strongly shifted toward zero relative to a
shuffle-based null model (Kolmogorov–Smirnov \(p \approx 4 \times 10^{-13}\),
Mann–Whitney \(p \approx 3 \times 10^{-13}\), Cliff’s \(\Delta \approx 0.32\)).  
Integer participation is not uniform: families such as
iron-based, oxide, heavy-fermion and molecular superconductors show systematically
smaller \(|\delta|\) than high-pressure phases and superfluid helium.  
The calibrated base frequencies form a smooth trend across families, with iron-based compounds near
\(f_{\text{base}} \sim 1\), oxide and heavy-fermion families near
\(f_{\text{base}} \sim 2 - 2.5\), and classical type-I metals around
\(f_{\text{base}} \sim 4.5 - 5\).

Structural noise matters. We quantify it via a per-family \(z\)-score
\(Z_\xi\) of the predicted \(T_c\) perturbation, and find a positive rank correlation
between noise and distance to integer locking
(Spearman \(\rho \approx 0.34, p \approx 6 \times 10^{-4}\)).  
When we divide the dataset into an “almost integer” subset (lowest 20% in \(|\delta|\) )
and the rest, the almost-integer group shows substantially reduced structural noise
(Cliff’s \(\Delta \approx -0.47\)). Taken together, these results suggest that
integer participation in the corrected Mother Frequency is a genuine feature of the
data, that it varies in a structured way across families, and that it is favored in
low-noise structural environments.

We interpret this as a complementary layer to the prime-space fingerprints reported
in earlier work: the discrete exponents on \(\{2,3,5,7\}\) organize how families
occupy a lattice of rational scales, while integer participation organizes how
strongly each material engages a common oscillatory backbone once structural noise
is factored out. Whether this reflects a deeper delayed-oscillator mechanism or
simply regularities in curated experimental data remains an open question.

---

## 1. Introduction

Superconductors and superfluids appear diverse at the microscopic level, yet display
surprisingly simple patterns when their macroscopic energy scales are expressed in
terms of a “Mother Frequency”
\(F_m = \Theta_D / T_c\),

where \(\Theta_D\) is a Debye temperature (or analogous phononic scale) and \(T_c\)
is the critical temperature. In previous work we showed that, after a universal
two-parameter correction, different families of superconductors and superfluid helium
cluster into robust prime-space fingerprints: products and ratios of small primes
\(\{2,3,5,7\}\) with family-dependent exponents. That study was deliberately
phenomenological: we did not attempt to derive those primes from a microscopic model,
but treated them as a coarse-grained grammar inspired by delayed-oscillator networks.

One major gap remained. The prime-space analysis compressed ratios between energy
scales, but did not address how strongly each material participates in an
underlying oscillatory backbone, nor how structural disorder and pressure reshape
that participation. Intuitively, a material whose Mother Frequency fits an integer
number of “jumps” of a base oscillation should be more robust to noise than one that
lands halfway between integers. Conversely, structural noise and pressure should
tend to smear out commensurabilities unless some discrete topology pushes the system
back toward low-order locks.

In this work we take a step toward quantifying that picture. We introduce an
integer participation model for the Mother Frequency, in which each material is
assigned a participation number
\(N = F_m^* / f_{\text{base}}\),
and we ask whether the corrected \(N\) are unusually close to integers compared to
simple null models. Here \(F_m^*\) is a structurally corrected Mother Frequency,
and \(f_{\text{base}}\) is a base frequency calibrated globally and per family by a
robust criterion.

The key questions are:

- **Integer participation:** Are the corrected participations \(N\) significantly
  closer to integers than expected from shuffled or continuous null models?

- **Family structure:** Does the strength of integer locking vary systematically
  across superconducting families and superfluid helium?

- **Structural noise:** Is strong integer locking associated with lower structural
  noise in a quantitative way?

We will see that the answer is “yes” to all three, with moderate but consistent
effect sizes.

---

## 2. Data and structural noise

### 2.1 Materials and families

We work with essentially the same curated dataset used in the prime-space study:
about 250 entries covering elemental type-I and type-II superconductors, binary and
molecular compounds, heavy-fermion and iron-based families, high-pressure
superconductors, and superfluid \(^4\)He at several pressures. Each entry carries:

- material name and family label (e.g. SC_Binary, SC_IronBased, Superfluid),
- a sub-network label (single, sigma, pi, pressure-specific modes),
- critical temperature \(T_c\),
- a Debye-like scale \(\Theta_D\) derived from phonon or thermodynamic data.

These are the same fields used to define the Mother Frequency
\(F_m = \Theta_D / T_c\)
in the original analysis.

### 2.2 Structural noise predictor

To account for structural and pressure-induced disorder we introduce a scalar
structural noise parameter \(\xi\) per material, obtained from an independent
pipeline that combines:

- local coordination features (e.g. effective valence, packing),
- pressure tags and simulated pressure response,
- topological descriptors of the superconducting “cluster” (coordination, branching).

The details of this predictor are not crucial for what follows; we only require that:

- it is trained or tuned independently of the integer participation analysis,
- it is evaluated uniformly across families,
- and it admits per-family statistics (mean and standard deviation).

For each family \(k\) we compute a standardized noise score

\[
Z_{\xi,i} = \frac{\xi_i - \mu_{\xi,k}}{\sigma_{\xi,k}},
\]

where \(\mu_{\xi,k}\) and \(\sigma_{\xi,k}\) are the family mean and standard
deviation. This allows us to compare the relative noise level of materials across
families on a common scale.

### 2.3 Noise-corrected Mother Frequency

Using the structural noise predictor we define an ideal critical temperature

\[
T_{c,\text{ideal}} = T_c (1 + \xi),
\]

which corresponds to the value expected if structural noise were “neutralized”
according to the model. The corrected Mother Frequency is then

\[
F_m^* = \frac{\Theta_D}{T_{c,\text{ideal}}}.
\]

For comparison we keep the raw Mother Frequency
\(F_m = \Theta_D / T_c\), but all
quantitative results below are based on \(F_m^*\).

---

## 3. Integer participation model

### 3.1 Base frequency calibration

For each material we want to express the corrected Mother Frequency as an effective
number of base “jumps”

\[
N_i = \frac{F_{m,i}^*}{f_{\text{base},k(i)}},
\]

where \(k(i)\) is the family of material \(i\) and \(f_{\text{base},k}\) is a
family-dependent base frequency. In a truly locked system the \(N_i\) would be
integers.

We estimate \(f_{\text{base},k}\) by minimizing a robust loss:

\[
L(f) = \text{median}_i\,|N_i - \text{round}(N_i)| + \lambda\,\text{mean}_i\,N_i,
\]

with \(\lambda\) a small regularization constant that softly prefers solutions with
smaller typical \(N\) when several harmonics tie. This avoids over-emphasizing rare
outliers and disfavors trivial rescalings such as \(N \to 10N\).

We explore two modes:

- a global calibration with a single \(f_{\text{base}}\) shared by all families,
- a per-family calibration with an independent \(f_{\text{base},k}\) for each
  of the nine major families.

In addition we test two simple hypotheses for the target frequency:

- \(H_1: F_{\text{target}} = F_m^*\),
- \(H_2: F_{\text{target}} = F_m^* / 2\),

motivated by the possibility that the Mother Frequency may represent either the
fundamental or a second harmonic of a more primitive oscillation. In practice the
\(F_m^*\) hypothesis provides equal or lower loss in almost all cases, and we use
it as default in the figures.

For each material we record the participation number \(N_i\), the distance to the
nearest integer

\[
\delta_i = |N_i - \text{round}(N_i)|,
\]

and several summary statistics described below.

### 3.2 Null models

To assess whether the observed integer participation is non-trivial we build simple
null models that preserve some aspects of the data but destroy any coherent locking
between \(\Theta_D\) and \(T_c\).

Our main null model is a shuffle:

- \(\Theta_D\) values are randomly permuted within the dataset (or within families),
- the structural noise correction and base-frequency calibration are repeated,
- and the resulting distances \(|\delta|_{\text{null}}\) are recorded.

Repeating this procedure many times yields an empirical null distribution for any
statistic of interest (e.g. median \(|\delta|\)). In addition we experimented with
gamma/log-normal fits to the continuous \(F_m^*\) distribution; these gave similar
qualitative results, so we focus on the shuffle null in what follows.

### 3.3 Statistical measures

We use three complementary measures to quantify integer locking:

- **Distance to integer:** the distribution of \(|\delta|\) across all materials.
- **Family-level medians:** the median \(|\delta|\) per family as a measure of
  locking strength.
- **Participation histograms:** the distribution of \(N_i\) itself.

Empirical vs null distributions are compared using:

- the Kolmogorov–Smirnov test,
- the Mann–Whitney rank test,
- Cliff’s \(\Delta\) effect size.

We quantify the relationship between structural noise and integer locking via:

- Spearman rank correlation between \(|\delta_i|\) and \(Z_{\xi,i}\),
- Cliff’s \(\Delta\) comparing noise between the “almost-integer” group
  (lowest 20% in \(|\delta|\)) and the rest.

---

## 4. Results

### 4.1 Global integer participation vs null models

Figure 1a compares the distribution of corrected participations \(N_i\) with that
obtained from shuffle-based null models. While both real and null distributions are
broad, the empirical histogram shows pronounced accumulations near low integers
(\(N \approx 1,2,3\)) and around a higher cluster (\(N \approx 24-25\)) that are
much less visible in the shuffled data. This suggests that certain participation
numbers are preferentially occupied once structural noise is taken into account.

Figure 1b focuses on the distance to integer \(|\delta|\), plotted on a log-density
scale. The real data are consistently more concentrated near zero than the shuffled
null: the tail of almost-integer cases is enhanced, and the shoulder at
\(|\delta| \gtrsim 0.2\) is depleted. Formal tests confirm that this difference is
highly significant:

- KS test \(p \approx 3.7 \times 10^{-13}\),
- Mann–Whitney \(p \approx 2.7 \times 10^{-13}\),
- Cliff’s \(\Delta \approx 0.32\) (moderate effect size).

Thus, under very conservative null models that keep the one-point statistics of
\(\Theta_D\) and \(T_c\) but scramble their pairing, the observed Mother Frequencies
are too close to integer participation to be explained by chance alone.

### 4.2 Family-dependent locking strength

Figure 2 summarizes the distribution of \(|\delta|\) per family as boxplots,
clipped at \(|\delta| = 1\) for readability and ordered by the median.

Several patterns stand out:

- Oxide, iron-based and heavy-fermion families display the smallest medians
  (strongest integer locking), with typical \(|\delta|\) of order \(0.05 - 0.1\).
- Molecular superconductors also show relatively tight locking.
- Type-I and type-II elemental and binary superconductors have broader distributions,
  with medians around \(0.15 - 0.2\).
- High-pressure phases and superfluid helium show both larger medians and broader
  spreads, consistent with the intuition that these systems operate in more
  strongly perturbed structural environments.

In other words, integer participation is not a homogeneous global phenomenon: some
families sit close to the integer lattice, others hover at a moderate distance.

### 4.3 Base frequency by family

The calibrated base frequencies \(f_{\text{base},k}\) are shown in Figure 4. Each
bar represents the family median, and the dashed line marks the global median,
about \(f_{\text{base}} \approx 4.16\).

- Iron-based superconductors sit near \(f_{\text{base}} \sim 1.1\),
- heavy-fermion and oxide families cluster around \(2 - 2.5\),
- molecular and high-pressure systems lie near \(4.5\),
- type-I elemental superconductors reach \(f_{\text{base}} \sim 4.8\).

The ordering of families by base frequency is smooth rather than random, and only
weakly sensitive to technical choices (such as global vs per-family calibration and
the \(F_m\) vs \(F_m/2\) hypothesis). This supports the view that different
superconducting families occupy distinct “bands” of participation in the same
corrected Mother Frequency.

### 4.4 Structural noise vs integer locking

Figure 3a plots \(|\delta|\) versus the structural noise \(Z_{\xi,i}\), with points
colored by family. There is a clear trend: materials with higher structural noise
tend to sit farther from integer participation. A Spearman rank correlation yields
\(\rho \approx 0.34\) with \(p \approx 5.6 \times 10^{-4}\), indicating a moderate
but statistically significant association across the heterogeneous dataset.

To emphasize the effect size, we split the dataset into two groups:

- **“almost integer”**: the 20% of materials with the smallest \(|\delta|\),
- **“rest”**: the remaining 80%.

Figure 3b compares the distributions of \(Z_\xi\) for these two groups. The
almost-integer group exhibits systematically lower structural noise, with a strong
Cliff’s effect size \(\Delta \approx -0.47\) (negative because noise is lower in
the almost-integer set). In other words, materials whose corrected Mother Frequency
participation is closest to an integer also tend to live in structurally quieter
environments according to the independent noise model.

---

## 5. Discussion

The integer-participation analysis adds a new layer to the phenomenology of
superconductors and superfluid helium:

- **Non-trivial integer locking.**  
  After correcting for structural noise, the Mother Frequency prefers integer
  participation numbers significantly more than expected from shuffled null models.
  This is not an artifact of a single family or a handful of outliers: it is a
  global shift in the \(|\delta|\) distribution with moderate effect size.

- **Structured family differences.**  
  Families known to host complex or strongly correlated superconductivity
  (iron-based, heavy-fermion, oxides, some molecular systems) exhibit the strongest
  integer locking, while high-pressure phases and superfluid helium show weaker
  but still detectable locking. At the same time, the calibrated base frequencies
  organize families along a smooth band from \(f_{\text{base}} \sim 1\) to
  \(f_{\text{base}} \sim 5\).

- **Link to structural noise.**  
  Structural noise, estimated independently from topological and pressure
  descriptors, is not neutral: high-noise materials are pushed away from the
  integer lattice, whereas low-noise materials accumulate near integer
  participations. This is exactly what one would expect if a coherent oscillatory
  backbone were being continuously perturbed by structural disorder.

From the perspective of the delayed-oscillator framework, these observations suggest
a qualitative picture. The Mother Frequency \(F_m^*\) plays the role of a coarse
grained oscillation whose phase advance per “jump” should ideally be commensurate
with a discrete lattice of participation numbers. Structural noise and pressure
perturb that commensurability; in low-noise environments the system can relax toward
near-integer participation, while in high-noise environments it is smeared away from
the integer grid.

From a more conventional standpoint, the present work can be read as a purely
phenomenological statement: once we correct \(\Theta_D / T_c\) by an independent
structural-noise model, the resulting dimensionless ratios display:

- statistically significant preference for integer participation,
- reproducible family-specific base frequencies,
- and a quantitative coupling to a noise proxy.

These are all empirical facts about curated data, independent of whether the DOFT
framework eventually finds a detailed microscopic realization.

---

## 6. Outlook

Several obvious next steps remain:

- **External validation.**  
  Apply the integer participation pipeline to new superconductors and to additional
  superfluid systems (e.g. \(^3\)He, mixed phases) as data become available, checking
  whether their participations and base frequencies fall into the existing bands or
  populate new ones.

- **Refining the structural noise model.**  
  The current noise predictor is deliberately simple. Replacing it with more
  detailed microscopic or ab-initio estimates of disorder and pressure response
  would provide a sharper test of the noise–locking relationship.

- **Connection to prime-space fingerprints.**  
  The prime exponents and rational denominators from the previous study and the
  integer participation numbers from this work are complementary summaries of the
  same systems. A natural next step is to explore whether particular prime-space
  fingerprints correlate with specific participation bands, and whether this can
  be framed as a discrete topology of delayed oscillators.

- **Predictive tests.**  
  Ultimately, the value of this framework will hinge on its ability to make and
  pass prospective predictions, for example by flagging materials that are “too
  far” from integer participation given their structural noise, or by suggesting
  pressure or compositional changes that move a candidate material toward favorable
  participation bands.

For now, we view the present work as a second “delivery”: the first showed that a
simple prime-based grammar can organize superconducting families into stable
fingerprints; this one shows that, after correcting for structural noise, the Mother
Frequency itself prefers integer participation in a structured, family-dependent
way. Whether this ultimately points to a deeper oscillator-based description or
simply exposes hidden regularities in experimental compilations is a question that
future data—and perhaps new materials—will decide.
