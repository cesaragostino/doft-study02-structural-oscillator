\documentclass[
  aps,
  prb,
  twocolumn,
  superscriptaddress,
  floatfix,
  10pt
]{revtex4-2}

% ====== Paquetes básicos ======
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{siunitx}
\sisetup{detect-all}
\usepackage{graphicx}
\graphicspath{{figures/}} % <-- busca figuras en paper/figures/
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}

% ====== Macros ======
\newcommand{\Tc}{T_{\mathrm{c}}}
\newcommand{\Thetad}{\Theta_{\mathrm{D}}}
\newcommand{\Fm}{F_{\mathrm{m}}}
\newcommand{\Fmstar}{F_{\mathrm{m}}^{\ast}}
\newcommand{\Ncorr}{N_{\mathrm{corr}}}
\newcommand{\Nraw}{N_{\mathrm{raw}}}
\newcommand{\deltan}{\delta}
\newcommand{\fbase}{f_{\mathrm{base}}}
\newcommand{\kB}{k_{\mathrm{B}}}

% ====== Documento ======
\begin{document}

\title{Integer participation and structural noise in the mother frequency of superconducting clusters}

\author{Cesar Agostino}
\affiliation{Independent researcher, Argentina}

\date{\today}

% ====== Abstract ======
\begin{abstract}
We study whether the ``mother frequency'' 
\(\Fm = \Thetad / \Tc\)
of superconductors and superfluid helium exhibits integer locking once structural noise is removed.
Building on a previously proposed delayed-oscillator framework, we treat
\(\Fm\) as the effective participation of a mesoscopic cluster in a discrete
ladder of modes and ask whether the noise-corrected ratio can be written as
\(\Fmstar \approx \fbase\, N\) with integer \(N\).
Using a curated dataset of \(\Tc\), Debye temperature, and structural descriptors
for nine families (Type-I, Type-II, binary, oxides, molecular, iron-based, heavy-fermion, high-pressure,
and superfluid helium), we:

(i) learn a structural-noise model that predicts distortions of \(\Tc\) from
crystal and pressure features;

(ii) define an ``ideal'' transition temperature
\(\Tc^{\mathrm{ideal}} = \Tc(1+\xi)\) and mother frequency
\(\Fmstar = \Thetad / \Tc^{\mathrm{ideal}}\);

(iii) calibrate a family-dependent base frequency \(f_{\mathrm{base},k}\)
by minimizing the median distance of \(\Fmstar / f_{\mathrm{base},k}\) to the nearest integer.

Across the full dataset the corrected participation numbers
\(\Ncorr = \Fmstar / \fbase\) show a strong excess of near-integer values
relative to null models that shuffle \(\Thetad\) across materials.
The distribution of fractional distances
\(|\deltan| = |\Ncorr - \mathrm{round}(\Ncorr)|\) is significantly more
concentrated near zero than in the shuffled ensemble, with
Kolmogorov--Smirnov and Mann--Whitney tests giving
\(p \sim 10^{-13}\) and a medium Cliff's \(\Delta \approx 0.32\).
Families differ systematically in their preferred base frequency:
iron-based superconductors cluster around \(f_{\mathrm{base},k} \approx 1\),
while Type-I metals sit near \(f_{\mathrm{base},k} \approx 4.7\),
with superfluid helium and high-pressure hydrides occupying intermediate
harmonics.

We also find that materials with the strongest integer participation
(the lowest 20\,\% in \(|\deltan|\)) have substantially lower structural noise
than the rest of the dataset, with a large negative Cliff's
\(\Delta \approx -0.47\) in the noise \(z\)-score.
Overall, these results support a phenomenological picture in which
structural disorder acts as a vector of dissipation that pulls mother
frequencies away from a small discrete set of participation numbers.
Whether this reflects a deeper oscillator-based mechanism or simply exposes
hidden regularities in curated superconducting data remains an open
question.
\end{abstract}

\maketitle

% ====== 1. Introduction ======
\section{Introduction}
\label{sec:intro}

The critical temperature \(\Tc\) of a superconductor is usually understood
as the outcome of microscopic pairing mechanisms, phonon spectra, and
electronic structure.\cite{BCS1957,Tinkham1996}
In parallel, phenomenological approaches often exploit empirical
scalings and coarse-grained ratios such as \(\Thetad/\Tc\) to compare
materials across families.
The present work revisits one such ratio, the
``mother frequency'' \(\Fm = \Thetad/\Tc\), from the perspective of
delayed-oscillator networks and structural noise.

In a related study we showed that a simple locking grammar on the primes
\(\{2,3,5,7\}\), applied to Debye, electronic and thermal scales, yields
stable ``prime-space fingerprints'' for superconductors and superfluid
helium under a universal two-parameter correction law.
Here we ask a simpler but sharper question:
after accounting for structural noise, does \(\Fm\) behave as the
participation of a mesoscopic cluster in a discrete ladder of modes?
Concretely, can we write
\begin{equation}
    \Fmstar \approx f_{\mathrm{base},k} N,
\end{equation}
with \(N \in \mathbb{Z}^{+}\) and a family-dependent base
frequency \(f_{\mathrm{base},k}\)?

The analysis proceeds in three steps.
First, we build a structural-noise model that predicts deviations in
\(\Tc\) from simple cluster descriptors and residual pressures.
Second, we define a noise-corrected mother frequency
\(\Fmstar = \Thetad / \Tc^{\mathrm{ideal}}\), where
\(\Tc^{\mathrm{ideal}}\) is the transition temperature one would
observe in the absence of structural distortions.
Third, we calibrate a base frequency for each family by minimizing the
median distance of \(\Fmstar/f_{\mathrm{base},k}\) to the nearest
integer, and compare the resulting integer participation pattern to
several null models.

Our main findings are:

\begin{enumerate}
    \item The corrected participation numbers \(\Ncorr\) show a statistically
    significant excess of near-integer values relative to shuffled null
    ensembles, with medium effect sizes and \(p\)-values well below
    \(10^{-10}\).

    \item Different families occupy distinct base frequencies:
    iron-based superconductors are compatible with \(f_{\mathrm{base},k} \sim 1\),
    while Type-I metals prefer higher harmonics
    \(f_{\mathrm{base},k} \sim 4\text{--}5\).

    \item The strongest integer lockers (lowest 20\,\% in \(|\deltan|\))
    have substantially lower structural noise than the rest of the
    dataset, suggesting that disorder acts as a vector of dissipation
    that smears integer participation.

\end{enumerate}

We emphasize that the present construction is phenomenological.
We do not attempt to derive \(\Fm\) from a microscopic theory of delayed
oscillators, nor to identify \(\fbase\) with a specific collective mode.
Instead, we treat integer participation as an \emph{a priori}
hypothesis about how coherent clusters might arrange themselves in the
presence of noise, and test that hypothesis against curated data and
explicit null models.

% ====== 2. Data and structural noise ======
\section{Data and structural-noise model}
\label{sec:data}

\subsection{Dataset and families}

The starting point is the same curated dataset as in the previous
study, updated to version~v7 and limited to entries with consistent
\(\Tc\) and \(\Thetad\) values.
Each row corresponds to a material--subnetwork pair and includes the
following fields: material name, category
(SC\_TypeI, SC\_TypeII, SC\_Binary, SC\_Oxide, SC\_Molecular,
SC\_IronBased, SC\_HeavyFermion, SC\_HighPressure, Superfluid),
subnetwork label (single, \(\sigma\), \(\pi\), etc.), transition
temperature \(\Tc\) (in kelvin), Debye temperature \(\Thetad\),
and a set of structural descriptors.

These descriptors include coarse information about lattice packing,
coordination, residual pressure, and qualitative flags for structural
instabilities identified in the experimental literature.
They play a dual role: they parameterize structural noise in the
transition temperature, and provide a basis for the null models
discussed below.

\subsection{Mother frequency and raw participation}

For each entry we define the raw mother frequency
\begin{equation}
    \Fm = \frac{\Thetad}{\Tc}
\end{equation}
measured in natural units.
If one postulates a family-dependent base frequency
\(f_{\mathrm{base},k}\) for category \(k\), the simplest participation
number is
\begin{equation}
    \Nraw = \frac{\Fm}{f_{\mathrm{base},k}}.
\end{equation}
In the absence of structural noise we would expect \(\Nraw\) to be
close to an integer.
In practice, experimental uncertainties, disorder, and systematic
differences across families all contribute to spread \(\Nraw\) away
from the nearest integer.

\subsection{Structural noise and ideal transition temperature}

To disentangle structural effects from genuine integer locking we fit
a structural-noise model for the transition temperature.
For each material we write
\begin{equation}
    \Tc^{\mathrm{ideal}} = \Tc \,(1 + \xi),
\end{equation}
where \(\xi\) is a dimensionless noise term predicted from structural
features.
The model is fitted separately for each family using a robust
regression that downweights outliers and captures broad trends in how
crystal complexity, residual pressure, and known instabilities modify
\(\Tc\).

Once the model is trained, we obtain a predicted noise
\(\hat{\xi}\) for each material and define
\begin{equation}
    \Fmstar = \frac{\Thetad}{\Tc^{\mathrm{ideal}}}
            = \frac{\Thetad}{\Tc (1+\hat{\xi})}.
\end{equation}
By construction, \(\Fmstar\) is the mother frequency one would obtain
if the material could be adiabatically ``cleaned'' of structural
distortions while keeping its underlying cluster topology fixed.

To compare noise levels across families we compute
a standardized noise score
\begin{equation}
    z_{\xi} =
    \frac{\hat{\xi} - \mu_{\xi,k}}{\sigma_{\xi,k}},
\end{equation}
where \(\mu_{\xi,k}\) and \(\sigma_{\xi,k}\) are the mean and standard
deviation of \(\hat{\xi}\) within family \(k\).
Negative values of \(z_{\xi}\) correspond to cleaner-than-average
structures, positive values to more disordered ones.

% ====== 3. Integer participation pipeline ======
\section{Integer participation pipeline}
\label{sec:methods}

\subsection{Base frequency calibration}

For each family \(k\) we seek a base frequency \(f_{\mathrm{base},k}\)
such that
\(\Ncorr = \Fmstar / f_{\mathrm{base},k}\) is as close as possible to
integers.
Instead of minimizing a mean squared error, which is sensitive to
outliers, we minimize the median absolute distance to the nearest
integer:
\begin{equation}
    \mathcal{L}(f) =
    \mathrm{median}_i \left|
        \frac{\Fmstar_i}{f} - \mathrm{round}\!\left(
        \frac{\Fmstar_i}{f}
    \right)\right|.
\end{equation}
The search is constrained to a physically motivated range
\(0.5 \le f \le 5.0\) to avoid trivial high-harmonic solutions
with extremely small base frequencies.

In addition, a small penalty proportional to the mean participation
\(\langle \Ncorr \rangle\) is added to break degeneracies between
integer-scaled solutions:
\begin{equation}
    \tilde{\mathcal{L}}(f) = \mathcal{L}(f) +
    \lambda \,\langle \Ncorr \rangle,
\end{equation}
with \(\lambda \ll 1\).
This mild regularization prefers lower harmonics (smaller \(N\)) when
two choices of \(f\) yield similar integer alignment.

Figure~\ref{fig:fbase_by_family} summarizes the calibrated
\(f_{\mathrm{base},k}\) values for all families.
Iron-based superconductors cluster near \(\fbase \approx 1.1\),
whereas Type-I metals sit near \(\fbase \approx 4.8\), with superfluid
helium and high-pressure hydrides in between.

\subsection{Integer distance and almost-integer group}

Given the calibrated base frequencies, the corrected participation
number for each material is
\begin{equation}
    \Ncorr = \frac{\Fmstar}{f_{\mathrm{base},k}}.
\end{equation}
We define the absolute distance to the nearest integer as
\begin{equation}
    |\deltan| =
    \left| \Ncorr - \mathrm{round}(\Ncorr) \right|.
\end{equation}
Small values of \(|\deltan|\) indicate strong integer locking.

To probe the relationship between structural noise and integer
locking we define an ``almost-integer'' group consisting of the lowest
20\,\% of the \(|\deltan|\) distribution and compare it to the rest of
the dataset using Cliff's \(\Delta\) on the noise \(z\)-score
\(z_{\xi}\).

\subsection{Null models}

Two null models are used to assess the statistical significance of the
observed integer participation:

\begin{enumerate}
    \item \textbf{Shuffle null:} within each family, we randomly
    permute \(\Thetad\) across materials while keeping \(\Tc\) and the
    structural noise model fixed.
    This preserves the marginal distributions of \(\Thetad\) and
    \(\Tc\) but breaks any material-specific locking.

    \item \textbf{Continuous null:} for the global \(\Fmstar\)
    distribution we fit a smooth continuous distribution (Gamma or
    lognormal) and draw synthetic \(\Fmstar\) values from the fitted
    law, re-using the same \(f_{\mathrm{base},k}\).
\end{enumerate}

For each null realization we recompute \(\Ncorr\) and \(|\deltan|\) and
compare the resulting distributions to the real data using
Kolmogorov--Smirnov (KS) and Mann--Whitney (MW) tests, together with
Cliff's \(\Delta\).

% ====== 4. Results ======
\section{Results}
\label{sec:results}

\subsection{Global integer participation}

Figure~\ref{fig:hist_N_real_vs_shuffle} compares the distribution of
corrected participation numbers \(\Ncorr = \Fmstar/f_{\mathrm{base},k}\)
for the real data and the shuffle null ensemble.
Both are broadly supported on \(N \sim 1\text{--}40\), but the real
distribution shows enhanced weight near small integers
\(N \sim 1\text{--}4\) and a pronounced spike around
\(N \sim 24\text{--}25\), corresponding mainly to superfluid helium
and selected high-pressure hydrides.

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{fig01a_hist_N_real_vs_shuffle}
    \caption{%
        Integer participation numbers \(\Ncorr = \Fmstar/f_{\mathrm{base},k}\)
        for the real data (blue) and shuffle null model (orange).
        The real distribution exhibits enhanced weight near small integers
        and a distinct cluster around \(N \sim 24\text{--}25\) not reproduced
        by the null ensemble.
    }
    \label{fig:hist_N_real_vs_shuffle}
\end{figure}

A more direct view of integer participation is shown in
Fig.~\ref{fig:hist_delta_real_vs_shuffle}, where we compare the
distribution of \(|\deltan|\) for real and shuffled data.
On a logarithmic density scale the real data are clearly more
concentrated near \(|\deltan| = 0\).
KS and MW tests yield \(p \approx 3.7 \times 10^{-13}\) and
\(p \approx 2.7 \times 10^{-13}\), respectively, with a medium
Cliff's \(\Delta \approx 0.32\).
Thus the probability that the observed excess of near-integer
participation arises from random reshuffling of Debye temperatures is
astronomically small.

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{fig01b_hist_delta_real_vs_shuffle}
    \caption{%
        Distribution of fractional distances to the nearest integer
        \(|\deltan| = |\Ncorr - \mathrm{round}(\Ncorr)|\) for the real
        data (blue) and shuffle null model (orange), plotted on a
        logarithmic density scale.
        The real distribution is significantly more concentrated near
        zero, with KS and MW \(p\)-values \(\sim 10^{-13}\) and
        Cliff's \(\Delta \approx 0.32\).
    }
    \label{fig:hist_delta_real_vs_shuffle}
\end{figure}

\subsection{Family-dependent locking strength}

Integer participation is not uniform across families.
Figure~\ref{fig:delta_by_family} shows box plots of \(|\deltan|\)
(clipped at 1.0 for readability) grouped by family and ordered by the
median.
Type-I and molecular superconductors exhibit the tightest locking,
with medians \(\lesssim 0.1\), followed by iron-based and heavy-fermion
materials.
Superfluid helium and high-pressure hydrides show broader
distributions, consistent with their more extreme structural and
pressure environments.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{fig02_delta_by_family}
    \caption{%
        Per-family distribution of \(|\deltan|\), the absolute distance
        to the nearest integer, clipped at 1.0 for visibility.
        Families are ordered by median \(|\deltan|\).
        Type-I and molecular superconductors show the tightest integer
        locking, whereas high-pressure and superfluid families are more
        broadly distributed.
    }
    \label{fig:delta_by_family}
\end{figure*}

The corresponding base frequencies are summarized in
Fig.~\ref{fig:fbase_by_family}.
Families organize along a discrete ladder:
iron-based superconductors near \(\fbase \approx 1.1\),
Type-II, oxides and heavy fermions near \(\fbase \approx 2.3\text{--}2.5\),
superfluid helium and binaries near \(\fbase \approx 4.0\text{--}4.1\),
and Type-I metals near \(\fbase \approx 4.8\).
The global median across families is \(\fbase \approx 4.16\),
close to the value obtained in the original prime-space fingerprint
study.

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{fig04a_fbase_by_family}
    \caption{%
        Calibrated base frequency \(\fbase\) per family, shown as the
        median of the best-fit values and ordered by that median.
        Iron-based superconductors sit near \(\fbase \approx 1.1\),
        whereas Type-I metals prefer higher harmonics around
        \(\fbase \approx 4.8\).
        The dashed line marks the global median
        \(\fbase \approx 4.16\).
    }
    \label{fig:fbase_by_family}
\end{figure*}

\subsection{Structural noise vs integer locking}

To probe the link between structural noise and integer participation
we compare \(|\deltan|\) to the standardized noise score \(z_{\xi}\).
Figure~\ref{fig:delta_vs_noise_scatter} shows a scatter plot of
\(|\deltan|\) vs.\ \(z_{\xi}\) for a representative subset of
materials, color-coded by family.
There is a clear trend: cleaner structures (negative \(z_{\xi}\))
tend to have smaller \(|\deltan|\), whereas disordered ones cluster at
higher \(|\deltan|\).
The Spearman rank correlation is
\(\rho \approx 0.34\) with \(p \approx 5.6 \times 10^{-4}\).

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{fig03a_delta_vs_noise_scatter}
    \caption{%
        Scatter plot of \(|\deltan|\) vs.\ structural-noise score
        \(z_{\xi}\), color-coded by family.
        Cleaner structures (negative \(z_{\xi}\)) tend to show stronger
        integer locking (smaller \(|\deltan|\)).
        The Spearman rank correlation is
        \(\rho \approx 0.34\) with \(p \approx 5.6 \times 10^{-4}\).
    }
    \label{fig:delta_vs_noise_scatter}
\end{figure}

A more direct comparison is shown in
Fig.~\ref{fig:noise_almost_integer_vs_rest}, where we contrast the
noise distribution of the almost-integer group (lowest 20\,\% in
\(|\deltan|\)) with that of the remaining materials.
The almost-integer group is shifted to lower noise, with median
\(z_{\xi} \approx -0.9\) compared to \(\approx 0.45\) for the rest,
and a large negative Cliff's \(\Delta \approx -0.47\).
This supports the interpretation of structural noise as a vector of
dissipation that smears discrete participation.

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{fig03b_noise_almost_integer_vs_rest}
    \caption{%
        Structural-noise score \(z_{\xi}\) for the almost-integer group
        (lowest 20\,\% in \(|\deltan|\)) and for the rest of the
        dataset.
        The almost-integer group is substantially cleaner, with a large
        negative Cliff's \(\Delta \approx -0.47\).
    }
    \label{fig:noise_almost_integer_vs_rest}
\end{figure}

% ====== 5. Discussion and outlook ======
\section{Discussion and outlook}
\label{sec:discussion}

The analysis above shows that once structural noise is explicitly
modeled and corrected, the mother frequency
\(\Fm = \Thetad/\Tc\) of superconductors and superfluid helium
exhibits a robust tendency to organize into near-integer
participation numbers.
This tendency is statistically significant relative to shuffle-based
null models and varies systematically across families.

From a delayed-oscillator perspective, one can view the base
frequencies \(f_{\mathrm{base},k}\) as effective resonant ladders for
different cluster topologies, with structural noise acting as a vector
of decoherence that pushes participation away from exact integers.
Iron-based superconductors, with \(f_{\mathrm{base},k} \sim 1\), behave
as if they were sitting close to a fundamental mode, whereas
Type-I metals appear to operate on higher harmonics
\(\sim 4\text{--}5\).
Whether this reflects genuine differences in underlying mesoscopic
oscillators, or simply tracks regularities in how lattice stiffness and
electron--phonon coupling co-vary across families, remains to be seen.

From a more conservative viewpoint, the results can be read as a
coarse-grained statistical statement:
after accounting for structural noise, the ratio \(\Thetad/\Tc\) is
not randomly distributed but instead shows a distinct excess of
near-integer values, with family-dependent base frequencies.
This is already non-trivial, as the null models preserve the marginal
distributions of \(\Thetad\) and \(\Tc\) and still fail to reproduce
the observed locking.

The present work should be viewed as a first delivery.
We have provided an explicit pipeline, open data, and a set of
sanity checks that can be refined or falsified as more materials and
better structural descriptors become available.
Several directions for future work are clear:

\begin{itemize}
    \item Incorporating dynamical data (e.g.\ phonon linewidths,
    pump--probe measurements) to test whether strongly locked materials
    also show distinctive coherence times.

    \item Extending the analysis to unconventional superconductors not
    covered in the present dataset and to other quantum fluids.

    \item Connecting the phenomenological base frequencies
    \(f_{\mathrm{base},k}\) to explicit models of delayed oscillators
    with memory and dissipation.\cite{Mori1965,Zwanzig2001}
\end{itemize}

Whether integer participation ultimately points to a deeper
oscillator-based description or simply exposes hidden regularities in
curated experimental data is a question that future observations will
decide.
For now, the main empirical statement is simple:
once structural noise is cleaned away, superconducting clusters and
superfluid helium prefer to lock their mother frequencies onto a
discrete ladder of participation numbers rather than wandering freely
in the continuum.

% ====== Acknowledgements ======
\begin{acknowledgments}
The author thanks the open-source and open-data communities for
making this work possible, and acknowledges intensive use of modern
language models for code refactoring, statistical sanity checks, and
editing assistance.
All modeling decisions, data curation and interpretations are the
responsibility of the author.
\end{acknowledgments}

% ====== Bibliography ======
\bibliographystyle{apsrev4-2}
\bibliography{mainNotes}

\end{document}
