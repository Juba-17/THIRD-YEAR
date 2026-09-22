# Statistical Inference for Artificial Intelligence
## DSAI 307 — Master Study Guide: Point Estimation (Lectures 1 & 2)

**Instructor:** Dr. Rasha Mandouh · Zewail City of Science, Technology and Innovation
**Primary source:** Lectures 01–02 slide content · **Textbook support:** Casella & Berger, DeGroot & Schervish, Wasserman, Rice, Murphy, Bishop, Hastie–Tibshirani–Friedman

> **Labels used throughout**
> **From the Lecture** — content taken directly from the slides
> **Textbook Expansion** — deeper/complementary material from a textbook, clearly attributed
> **Additional Explanation** — bridging explanation not explicit in the slides but needed for understanding
> **AI/ML Connection** — link to machine learning
> **Common Misconception** — a mistake students typically make
> **Important Note** — a caution or clarification

---

## Overview

This guide teaches **Point Estimation**, the first of the three pillars of statistical inference in this course (Point Estimation → Interval Estimation → Hypothesis Testing). It answers one central question: *given a random sample, how do we produce a single "best guess" number for an unknown population parameter, and how do we judge whether that guess-producing rule is any good?*

The guide is built entirely from Lectures 1 & 2. It does **not** invent content on hypothesis testing, confidence intervals, or Bayesian inference beyond what the lecture itself uses to situate point estimation within the larger field — those are explicitly **future lectures** in this course, and are only sketched here for orientation, never derived.

## Learning Objectives

By the end of this guide you should be able to:

1. Distinguish population, sample, parameter, statistic, estimator, and estimate.
2. Explain why a statistic is a random variable with its own sampling distribution.
3. Define and apply unbiasedness, efficiency, and Mean Squared Error (MSE).
4. Derive the bias–variance decomposition of MSE from first principles.
5. Use relative efficiency to compare two estimators (biased or unbiased).
6. Construct Method-of-Moments (MoM) estimators for one- and two-parameter distributions.
7. Compute point estimates numerically from raw data.
8. Recognize common exam traps and misconceptions in point estimation.
9. Explain how these ideas reappear in AI/ML as the bias–variance tradeoff and moment-matching initialization.

## Prerequisites

**Additional Explanation.** The lecture assumes familiarity with:
- Basic probability: random variables, expectation $E(\cdot)$, variance $V(\cdot)$.
- Discrete vs. continuous distributions (pmf/pdf).
- Summation notation and basic algebra (solving simultaneous equations, for Method of Moments with 2+ parameters).
- The idea of a "random sample" $X_1,\ldots,X_n$ — independent, identically distributed (i.i.d.) draws from a population.

If any of these feel shaky, review expectation/variance identities in §5 before proceeding (Casella & Berger, Ch. 2; DeGroot & Schervish, Ch. 4).

---

## Statistical Foundations

### The Inference Pipeline

**From the Lecture.**

```text
Population
    ↓
Statistical Model  (assumed distribution family, e.g. Normal(μ, σ²))
    ↓
Random Sample       X₁, X₂, …, Xₙ
    ↓
Observed Data        x₁, x₂, …, xₙ
    ↓
Statistic            T(X₁,…,Xₙ)
    ↓
Inference
    ├── Point Estimation      ◄── THIS GUIDE
    ├── Interval Estimation   (future lecture)
    └── Hypothesis Testing    (future lecture)
    ↓
Conclusion about the Population
```

**Intuition.** Everything above "Inference" is *probability*: given a known model, what data will it produce? Everything from "Inference" downward is the reverse problem: given observed data, what can we say about the unknown model that produced it? Point estimation is the simplest form of this reversal — collapse all the sample information into **one number** standing in for the unknown parameter.

**Textbook Expansion (Casella & Berger, Ch. 7).** This is the classical **frequentist** paradigm: θ is fixed but unknown, and all randomness lives in the sampling mechanism. This differs from the **Bayesian** paradigm (§ Bayesian Inference below), where θ itself is treated as a random variable with a prior distribution — not covered in this lecture, but useful to know the field has two major schools.

### Core Distinctions (must never be conflated)

| Term | Meaning |
|---|---|
| **Population** | The entire group of items/individuals of interest. |
| **Sample** | A random subset of the population actually observed. |
| **Parameter (θ)** | A fixed, usually unknown numerical property of the *population* (e.g. μ, σ², p). |
| **Statistic** | Any computable function of sample data alone, e.g. $T(X_1,\ldots,X_n)$ — contains no unknown parameters. |
| **Estimator ($\hat\Theta$)** | A statistic *used as a rule* for estimating a parameter — a random variable **before** data is observed. |
| **Estimate ($\hat\theta$)** | The specific number $\hat\Theta$ takes **after** data is observed. |
| **Random variable vs. observed value** | $X_i$ (capital) is the not-yet-observed random draw; $x_i$ (lowercase) is its realized numeric value. |
| **Population distribution vs. sampling distribution** | The population distribution describes individual $X_i$'s; the sampling distribution describes the statistic $T(X_1,\ldots,X_n)$ across repeated samples — these are **never the same object**. |

**Common Misconception.** Confusing the population distribution (what one data point looks like) with the sampling distribution (what the *statistic*, like $\bar X$, looks like across repeated samples). The sampling distribution is almost always less spread out than the population distribution — e.g. $V(\bar X) = \sigma^2/n < \sigma^2 = V(X)$ for $n>1$.

---

## Main Concepts

### 1. Statistical Inference: Estimation vs. Hypothesis Testing

**Definition (From the Lecture).**
> **Statistical inference** consists of those methods by which one makes inferences or generalizations about a population.

```text
                Statistical Inference
                        │
          ┌─────────────┴─────────────┐
          ▼                            ▼
      Estimation              Tests of Hypotheses
          │
    ┌─────┴─────┐
    ▼           ▼
 Point       Interval
Estimation   Estimation
```

**Example — Estimation.** A candidate for public office samples 100 eligible voters to estimate the true proportion favoring him. The sample fraction estimates the unknown population proportion; the sampling distribution of the proportion tells us how accurate that estimate is. → This is **estimation**.

**Example — Hypothesis Testing.** We ask whether floor wax brand A is more scuff-resistant than brand B. We state a hypothesis and, after testing, **accept or reject** it — we are not estimating a value, we are making a binary decision. → This is **hypothesis testing**.

**Why It Matters.** The question shapes the tool: "What is it?" (estimation) needs an estimator and a precision measure; "Is it true?" (hypothesis testing) needs a decision rule with controlled error rates (Type I/II errors — a **future lecture**). Both rely on the same underlying machine: sampling distributions.

---

### 2. Point Estimation — Formal Setup

**Motivating Example (From the Lecture).** Let **μ** denote the true average breaking strength of wire connections used in bonding semiconductor wafers. A random sample of *n* = 10 connections is tested, giving observed strengths $x_1,\ldots,x_{10}$. The sample mean is used to draw a conclusion about μ.

**Objective.**
> The objective of point estimation is to select a **single number**, based on sample data, that represents a sensible value for **θ** (a general parameter).

**Definition.**
> A **point estimate** of a population parameter θ is a single numerical value $\hat\theta$ of a statistic $\hat\Theta$. The statistic $\hat\Theta$ is called the **point estimator**.

**Mathematical Meaning.** If we write $\hat\Theta = T(X_1,\ldots,X_n)$:
- $\theta$ — the unknown, fixed population parameter (e.g. true mean μ). Unknown, not random.
- $\hat\Theta$ — the **estimator**: the formula $T$ applied to the *random* sample $X_1,\ldots,X_n$. Because the $X_i$ are random, $\hat\Theta$ is random too, **before** data is collected.
- $\hat\theta = T(x_1,\ldots,x_n)$ — the **estimate**: the specific number obtained **after** plugging in the observed data $x_1,\ldots,x_n$. Fixed, not random.
- $T$ is called a *statistic* because it is a function of the sample alone and involves no unknown parameters — you could compute it the instant you have data, without knowing θ.

**Intuition.** Think of $\hat\Theta$ as a *recipe* ("add the numbers, divide by how many") and $\hat\theta$ as the *dish* you get after actually cooking with today's ingredients (today's sample). The recipe is fixed and reusable; the dish varies every time you cook.

**Key relationships (From the Lecture):**
- $\bar x$ (value of $\bar X$) → point estimate of the population mean **μ**.
- $\hat p = x/n$ → point estimate of the true proportion **p** in a binomial experiment.

**Textbook Expansion (Casella & Berger).** More generally, an estimator is *any* function $W(X_1,\ldots,X_n)$ of the sample not depending on unknown parameters — this is what makes it a statistic in the first place. Point estimation theory studies how to construct such functions (§ Method of Moments) and how to rank competing choices (§ Properties of Estimators below).

#### Worked Examples

**Example — Battery Lifetimes (point estimate of μ).** μ = true average battery lifetime. Sample *n*=3: $x_1=5.0, x_2=6.4, x_3=5.9$ hours.
$$\bar x = \frac{5.0+6.4+5.9}{3} = 5.77$$
**Interpretation:** 5.77 hours is our single best guess for μ from this sample.

**Example — Car Bumper Damage (point estimate of a proportion).** 25 crashes at 10 mph; *X* = number with no visible damage; *p* = true proportion with no damage; observed *x*=15.
$$\hat p = \frac{X}{n}\ (\text{estimator}) \qquad \hat p = \frac{15}{25}=0.60\ (\text{estimate})$$

**Example — Sample Mean from Normal Data.** *n*=10: $12.8, 9.4, 8.7, 11.6, 13.1, 9.8, 14.1, 8.5, 12.1, 10.3$.
$$\bar x = \frac{110.4}{10} = 11.04$$
(This dataset is reused later to illustrate unbiasedness.)

---

### 3. Sampling and Sampling Distributions

**From the Lecture.** Before data are collected, the observations are random variables $X_1,\ldots,X_n$. **Any function of them (any statistic) is therefore also a random variable.**

```text
Population
   │
   ├── Sample 1 of n=10  →  x̄ = 10.2
   ├── Sample 2 of n=10  →  x̄ = 10.4
   │        ⋮                  ⋮
```

Each new sample yields a (potentially) different $\bar x$. This is why $\bar X$ and $S^2$ are called **statistics**: random variables with their own probability distribution — the **sampling distribution**.

**Simulation-Based Intuition.**
```text
Population
   ↓
Draw a sample of size n
   ↓
Compute the statistic (e.g. x̄)
   ↓
Repeat many times
   ↓
The histogram of all those x̄ values IS the sampling distribution
```
Imagining this repeated-sampling process is the most reliable way to *feel* why an estimator's mean and variance are meaningful quantities, even though in real life you only ever get to draw one sample.

**Why It Matters.** Since $\hat\Theta$ is a random variable, it has a mean $E(\hat\Theta)$ and variance $V(\hat\Theta)$ — exactly the two numbers that let us judge whether one estimator is "better" than another (§ Properties of a Good Estimator below).

**Textbook Expansion (Rice, Ch. 5 — Asymptotic preview).** For $\bar X$ from an i.i.d. sample with population mean μ and variance $\sigma^2$:
$$E(\bar X) = \mu \qquad V(\bar X) = \frac{\sigma^2}{n}$$
As $n\to\infty$, $V(\bar X)\to 0$ — the sampling distribution concentrates tightly around μ, which is the intuition behind the **Law of Large Numbers**. By the **Central Limit Theorem**, the sampling distribution of $\bar X$ also becomes approximately Normal regardless of the population's shape, for large *n*. **Important Note:** this is an *asymptotic* (large-sample) result — "approximately Normal" is never the same as "exactly Normal" for finite *n*. Full asymptotic theory (convergence in probability/distribution, CLT proofs) is outside this lecture's scope and belongs to a later course unit.

### 4. Parameters We Commonly Estimate

**From the Lecture.** We often need to estimate:
1. The mean **μ** of a single population
2. The variance **σ²** (or σ) of a single population
3. The proportion **p** of items in a class of interest
4. The difference in means of two populations, **μ₁ − μ₂**
5. The difference in two proportions, **p₁ − p₂**

| Parameter | Point Estimate | Description |
|---|---|---|
| μ | $\hat\mu=\bar x$ | Sample mean |
| σ² | $\hat\sigma^2=s^2$ | Sample variance |
| p | $\hat p = x/n$ | Sample proportion |
| μ₁−μ₂ | $\bar x_1-\bar x_2$ | Difference of sample means (independent samples) |
| p₁−p₂ | $\hat p_1-\hat p_2$ | Difference of sample proportions (independent samples) |

---

## Properties of a Good Estimator

### 5. Unbiasedness

**The Central Question (From the Lecture).** What properties make one estimator preferable to another? For μ we could use the sample **mean** $\bar X$ or the sample **median** $\tilde X$ — we need objective criteria, decided *before* seeing data (we can never check against the true θ in practice).

**Illustrating the Problem.**
- Sample 1: 2, 5, 11 (true μ = 4, known only for illustration): $\bar x=6$, $\tilde x=5$ → median closer this time.
- Sample 2: 2, 6, 7 (same population): $\bar x=5$, $\tilde x=6$ → mean closer this time.

**Additional Explanation.** We cannot judge estimators case-by-case since we never know the true μ in real applications. We need a property that holds *on average, across all possible samples* — this motivates unbiasedness.

**Definition (From the Lecture).**
> Let $\hat\Theta$ be an estimator of θ. We would like the sampling distribution of $\hat\Theta$ to have a mean equal to the parameter it estimates. Such an estimator is **unbiased**:
> $$\mu_{\hat\Theta} = E(\hat\Theta) = \theta$$

**Mathematical Meaning.** Define **Bias** as $\text{Bias}(\hat\Theta) = E(\hat\Theta) - \theta$. Unbiased means $\text{Bias}(\hat\Theta)=0$. Here $E(\hat\Theta)$ is taken over the sampling distribution (over all possible samples of size *n*), not over one sample.

**Intuition.** On average — across the (conceptually infinite) population of all possible samples of size *n* — the estimator neither systematically over- nor under-shoots the truth. A single estimate can still be off; unbiasedness is a statement about the long-run average, not about any one sample.

**Assumptions.** Unbiasedness is defined relative to a *specific sample size n* and a *specific sampling scheme* (i.i.d. random sampling is assumed throughout this lecture); it says nothing about what happens for a single draw or under non-random sampling.

**Example — Multiple Unbiased Estimators of μ.** Using the *n*=10 dataset above:

| Statistic | Formula | Value |
|---|---|---|
| Sample mean | $\bar x$ | **11.04** |
| Sample median | $(10.3+11.6)/2$ | **10.95** |
| 10% trimmed mean | drop top/bottom 10%, average rest | **10.98** |

All three can be shown to be **unbiased estimators of μ** (for a symmetric population like the Normal).

> **Important Note.** Because the unbiased estimator is *not unique*, unbiasedness alone cannot crown a single "best" estimator — we need a second criterion: efficiency (§6).

**Example — MoM Estimator of *p* is Unbiased (Bernoulli).** $X_1,\ldots,X_n$ i.i.d. Bernoulli(*p*). Estimator: $\hat p = \frac1n\sum X_i = Y/n$, $Y=\sum X_i\sim\text{Binomial}(n,p)$.

**Derivation.** Since $E(Y)=np$ for a binomial random variable:
$$E(\hat p) = E\!\left(\frac{Y}{n}\right) = \frac1n E(Y) = \frac1n\cdot np = p\quad\checkmark$$
**Interpretation:** $\hat p = Y/n$ is unbiased for *p*.

**Limitations.** Unbiasedness is a *population-level* guarantee, not a per-sample one — one estimate can still be far from θ. An unbiased estimator can also have enormous variance, making individual estimates unreliable — motivating §7 (MSE).

**Common Misconception.** Assuming unbiasedness implies **consistency** (that the estimator converges to θ as $n\to\infty$), or vice-versa. These are related but logically independent properties; the lecture's dataset examples only address unbiasedness, not the large-sample behavior (consistency belongs to asymptotic theory, a future topic).

**AI/ML Connection.** In ML, we usually don't insist on unbiasedness — we optimize expected loss (which behaves like bias² + variance) directly, and often deliberately introduce bias (L2/L1 regularization, weight decay, shrinkage) to reduce variance and improve generalization. This is the statistical root of the ML **bias–variance tradeoff** used to explain overfitting/underfitting.

---

### 6. Efficiency (Variance)

**Definition (From the Lecture).**
> If $\hat\Theta_1$ and $\hat\Theta_2$ are both unbiased estimators of θ, we prefer the one whose sampling distribution has **smaller variance**. If $\sigma^2_{\hat\theta_1}<\sigma^2_{\hat\theta_2}$, then $\hat\Theta_1$ is **more efficient** than $\hat\Theta_2$.

**Intuition (Figure 1 in slides).** Picture two bell-shaped sampling distributions, both centered exactly at θ (both unbiased), but one is narrower/peakier. The peakier one gives estimates that cluster tightly around the truth — less likely to be far off on any given sample. That is efficiency.

**Three-Estimator Comparison.** Consider $\hat\Theta_1,\hat\Theta_2,\hat\Theta_3$ estimating the same θ:
- $\hat\Theta_1,\hat\Theta_2$ unbiased (centered at θ); $\hat\Theta_3$ **biased** (centered elsewhere) — disqualified on that basis alone.
- Between the two unbiased ones, $\hat\Theta_1$ has smaller variance → more efficient.
- **Overall best:** $\hat\Theta_1$ — unbiased *and* most efficient.

**Decision hierarchy so far:**
1. Prefer unbiased estimators.
2. Among unbiased estimators, prefer smallest variance.

**Textbook Expansion — Cramér–Rao Lower Bound (Casella & Berger, Ch. 7.3).** Among all unbiased estimators, there is a theoretical floor on the achievable variance, given by the Fisher Information $I(\theta)$:
$$V(\hat\Theta) \ge \frac{1}{n\,I(\theta)}$$
An unbiased estimator achieving this bound is called *efficient* in the strict sense. This is the rigorous foundation behind the lecture's informal "smaller variance is better" rule; the full derivation of $I(\theta)$ requires likelihood theory and is outside this lecture's scope.

**AI/ML Connection.** Efficiency parallels *sample efficiency* in ML: an algorithm reaching a target accuracy with fewer samples is more "efficient," directly analogous to a lower-variance estimator needing less data to pin down θ precisely.

---

### 7. Mean Squared Error (MSE)

**Why It Exists (From the Lecture).** Sometimes a **biased** estimator is unavoidable or even preferable (no unbiased estimator exists, or a biased one behaves better overall). We need one criterion capturing **both** bias and variance at once.

**Intuition (Figure 3 in slides).** A biased $\hat\Theta_1$ can have much smaller variance than an unbiased $\hat\Theta_2$, even though $\hat\Theta_1$ isn't centered exactly at θ. If the variance savings outweigh the bias penalty, the biased-but-tight estimator can be the better overall choice.

**Definition.**
$$\text{MSE}(\hat\Theta) = E\big[(\hat\Theta-\theta)^2\big]$$

**Derivation — Bias–Variance Decomposition (key result, show completely).**

Starting point: $\text{MSE}(\hat\Theta) = E[(\hat\Theta-\theta)^2]$.

Step 1 — insert and subtract $E(\hat\Theta)$ inside the square:
$$\text{MSE}(\hat\Theta) = E\Big[\big(\hat\Theta-E(\hat\Theta)\big)+\big(E(\hat\Theta)-\theta\big)\Big]^2$$

Step 2 — expand the square (this is $(a+b)^2 = a^2+2ab+b^2$ with $a=\hat\Theta-E(\hat\Theta)$ and $b=E(\hat\Theta)-\theta$):
$$= E\big[\hat\Theta-E(\hat\Theta)\big]^2 + 2\,E\big[(\hat\Theta-E(\hat\Theta))(E(\hat\Theta)-\theta)\big] + \big[E(\hat\Theta)-\theta\big]^2$$

Step 3 — the cross term vanishes: $b=E(\hat\Theta)-\theta$ is a constant (not random), so it factors out of the expectation, leaving $E[\hat\Theta-E(\hat\Theta)] = 0$ by definition of the mean:
$$2\,\big(E(\hat\Theta)-\theta\big)\cdot E\big[\hat\Theta-E(\hat\Theta)\big] = 2\,\big(E(\hat\Theta)-\theta\big)\cdot 0 = 0$$

Step 4 — what remains is exactly variance plus squared bias:
$$\text{MSE}(\hat\Theta) = \underbrace{E\big[\hat\Theta-E(\hat\Theta)\big]^2}_{=\,V(\hat\Theta)} + \underbrace{\big[\theta-E(\hat\Theta)\big]^2}_{=\,(\text{Bias})^2}$$

$$\boxed{\text{MSE}(\hat\Theta) = V(\hat\Theta) + (\text{Bias})^2}$$

**Interpretation.** MSE splits an estimator's total average squared error into two independent sources: how *spread out* it is (variance) and how *off-center* it is (squared bias). **Special case:** if $\hat\Theta$ is unbiased, Bias=0, so $\text{MSE}(\hat\Theta)=V(\hat\Theta)$ — consistent with §6, where unbiased estimators were compared by variance alone.

**Textbook Expansion.** This decomposition reappears throughout supervised learning theory, where expected test error of a prediction function decomposes into (squared) bias, variance, and irreducible noise — same algebra, different notation.

**AI/ML Connection.** This is *literally* the algebra behind the ML **bias–variance tradeoff**: a high-capacity model (e.g. a deep, unregularized network) tends to have low bias but high variance (overfits); a low-capacity model (e.g. linear regression on nonlinear data) has high bias but low variance (underfits). Regularization, ensembling, and early stopping all move along this same tradeoff curve to minimize an MSE-like generalization error.

**Common Misconception.** Writing "MSE = Variance" and forgetting the bias² term — this equality only holds in the unbiased special case.

---

### 8. Relative Efficiency

**Definition (From the Lecture).** For two estimators $\hat\Theta_1,\hat\Theta_2$ of the same θ (biased or unbiased):
$$\text{Relative efficiency of } \hat\Theta_2 \text{ to } \hat\Theta_1 = \frac{\text{MSE}(\hat\Theta_1)}{\text{MSE}(\hat\Theta_2)}$$

**Interpretation Rule.**
> If this ratio is **less than 1**, $\hat\Theta_1$ is the more efficient estimator (smaller MSE).

**Reading it correctly:**
$$\frac{\text{MSE}(\hat\Theta_1)}{\text{MSE}(\hat\Theta_2)} < 1 \iff \text{MSE}(\hat\Theta_1) < \text{MSE}(\hat\Theta_2) \implies \hat\Theta_1 \text{ wins}$$

**Common Misconception.** It is easy to reverse this direction under exam pressure — always translate the ratio back into "which MSE is smaller" before concluding.

**Worked Exercise — Calculator Brands (Proportion Estimation).** 20 students; brands owned: `T T H T C T T S C H / S S T H C T T T H T` (T = Texas Instruments).

Count T's: Row 1 → 5, Row 2 → 5, total = 10 out of *n*=20.
$$\hat p = \frac{10}{20} = 0.50$$
**Interpretation:** an estimated 50% of such students own a TI calculator.

---

## Methods of Point Estimation

### 9. Why We Need "Methods"

**From the Lecture.** Unbiasedness, efficiency, and MSE (§5–§8) tell us how to **judge** estimators — they say nothing about how to **construct** one in the first place. Two systematic construction methods:

1. **Method of Moments (MoM)** — equate population moments to sample moments.
2. **Maximum Likelihood Estimation (MLE)** — maximize the likelihood of the observed data. **[Marked in the lecture as the start of the next topic — the lecture only reaches the point of motivating MLE via Exercise §10.7 below; the full likelihood-function derivation, log-likelihood, and $\arg\max$ machinery are not covered in Lectures 1–2 and are therefore not derived in this guide, per the source-fidelity rule. See "Important Note" below.]**

**Important Note.** Because MLE's formal treatment (likelihood function $L(\theta;x)$, log-likelihood, $\hat\theta_{MLE}=\arg\max_\theta L(\theta;x)$, asymptotic properties) is not part of the supplied lecture, this guide does **not** derive it — doing so would risk presenting invented, un-sourced material as lecture content. The comparison table below reflects only what the lecture states about MLE *qualitatively*.

| Aspect | Method of Moments | Maximum Likelihood |
|---|---|---|
| General efficiency | Generally *less* efficient | Generally **preferable** |
| Ease of computation | Often easier, closed-form | Can require solving harder optimization/likelihood equations |
| Unbiasedness | Can be unbiased | Can also be unbiased |

**AI/ML Connection.** MLE is the direct ancestor of how most ML models are trained: minimizing cross-entropy loss for classification, or MSE loss for regression under a Gaussian-noise assumption, are both special cases of maximizing likelihood — formally, $\arg\max_\theta L(\theta;x) \equiv \arg\min_\theta\big[-\log L(\theta;x)\big]$, i.e. maximizing likelihood is the same as minimizing negative log-likelihood. Method of Moments corresponds to simpler moment-matching techniques still used for quick parameter initialization (e.g. initializing mixture-model or GAN training). *(This connection is stated here only at the level the lecture supports; the full NLL/cross-entropy derivation belongs to a later ML course.)*

### 10. Method of Moments — Theory

**General Idea (From the Lecture).**
> Equate **population moments** (expected values) to the corresponding **sample moments**. Population moments are functions of the unknown parameters; solving the resulting equations yields the estimators.

**Formal Definition.**
> Let $X_1,\ldots,X_n$ be a random sample from $f(x)$ (discrete pmf or continuous pdf).
> - **k-th population moment:** $E(X^k)$, $k=1,2,\ldots$
> - **k-th sample moment:** $\dfrac1n\sum_{i=1}^n X_i^k$, $k=1,2,\ldots$

**First Moment — Simplest Case.**
- Population: $E(X)=\mu$. Sample: $\hat\mu = \frac1n\sum X_i = \bar X$. Equate: $\hat\mu=\bar X$.
> **The sample mean is the moment estimator of the population mean** — the fastest shortcut for any one-parameter distribution whose mean directly *is* that parameter (exponential mean=β, gamma mean=αβ, shifted-exponential mean=$1+\eta$, etc.).

**General Case (m Parameters).**
> Let $\theta_1,\ldots,\theta_m$ be *m* unknown parameters. The **moment estimators** $\hat\Theta_1,\ldots,\hat\Theta_m$ are found by equating the first *m* population moments to the first *m* sample moments and solving simultaneously.

**Problem-Solving Checklist — Method of Moments**
```text
1. Count the unknown parameters, m.
2. Write E(X), E(X²), …, E(X^m) as functions of the parameters.
3. Write the matching sample moments (1/n)ΣXᵢᵏ for k = 1,…,m.
4. Set population moment = sample moment for each k.
5. Solve the m×m system for the m unknown parameters.
6. Report the estimator(s); verify units/sign make sense.
7. If asked: check unbiasedness by computing E(estimator) directly.
```

**Assumptions.** MoM implicitly assumes the relevant population moments exist and are finite (it fails for heavy-tailed distributions like the Cauchy, where $E(X)$ doesn't exist), and that the sample is i.i.d. from the assumed family.

### 11. Method of Moments — Fully Worked Examples

**Example — Exponential Distribution (One Parameter).** $X_1,\ldots,X_n\sim\text{Exponential}(\lambda)$.
$$E(X)=\frac1\lambda=\bar x \implies \boxed{\hat\lambda=\frac1{\bar x}}$$

**Numerical — Electronic Module Failure Times.** Data (hrs), n=8: 11.96, 5.03, 67.40, 16.07, 31.50, 7.73, 11.10, 22.38.
$$\bar x = \frac{173.17}{8}=21.65 \qquad \hat\lambda = \frac1{21.65}=0.0462$$

**Example — Normal Distribution (Two Parameters: μ, σ²).**

Step 1 — population moments: $E(X)=\mu$, $E(X^2)=\mu^2+\sigma^2$.

Step 2 — equate: $\mu=\bar x$; $\mu^2+\sigma^2 = \frac1n\sum x_i^2$.

Step 3 — solve:
$$\hat\mu=\bar x \qquad \hat\sigma^2 = \frac{\sum(x_i-\bar x)^2}{n}$$

**Statistical Interpretation / Discussion (posed in lecture): is $\hat\mu$ unbiased?** Yes — $E(\bar X)=\mu$ always (§5). **Important Note:** $\hat\sigma^2$ here divides by *n*, not *n−1*; this is the MLE/MoM form and is in fact a **biased** estimator of $\sigma^2$ (it slightly underestimates on average). The unbiased version uses $n-1$ in the denominator — this contrast is a classic exam trap.

**Example — Gamma Distribution (Two Parameters: α, β).**

Step 1: $E(X)=\alpha\beta$, $E(X^2)=\alpha(\alpha+1)\beta^2$.

Step 2: $\bar X=\alpha\beta$; $\frac1n\sum X_i^2 = \alpha(\alpha+1)\beta^2$.

Step 3 (solve the system):
$$\hat\alpha = \frac{\bar X^2}{\frac1n\sum X_i^2-\bar X^2} \qquad \hat\beta = \frac{\frac1n\sum X_i^2-\bar X^2}{\bar X}$$

**Numerical (n=20 survival times):** given $\bar x=113.5$, $\frac1{20}\sum x_i^2=14{,}087.8$:
$$\hat\alpha = \frac{(113.5)^2}{14{,}087.8-(113.5)^2}=10.7 \qquad \hat\beta = \frac{14{,}087.8-(113.5)^2}{113.5}=10.6$$

**Example — Two-Parameter (Shifted) Exponential.** $X_i\sim\text{EXP}(1,\eta)$, mean $=1+\eta$.
$$\bar x = 1+\hat\eta \implies \boxed{\hat\eta = \bar x - 1}$$

**Example — Unbiasedness of the MoM Estimator for β (Exponential).** MoM estimator is $\bar X$; population mean is β, so $E(\bar X)=\beta$ ⟹ **unbiased.** ✓

**Exercise — Supermarket Checkout Arrivals (Bridge to MLE).** Data (min), n=14: 1,2,3,7,11,4,13,12,7,3,2,11,7,2 → sum=85. Model: $f(x)=\frac1\theta e^{-x/\theta}$ (mean θ).
$$\bar x = \frac{85}{14}\approx 6.071 \implies \boxed{\hat\theta\approx 6.07\text{ minutes}}$$
This exercise is the slide's explicit bridge into MLE (the next topic), which begins by reviewing $E(X)$ definitions and the distribution table below.

---

## Probability Distributions

### General Definition of the Mean (From the Lecture)
> For a random variable *X* with distribution $f(x)$:
> $$\mu=E(X)=\sum_x x\,f(x)\ \ (\text{discrete}) \qquad \mu=E(X)=\int_{-\infty}^{\infty}x\,f(x)\,dx\ \ (\text{continuous})$$

This is the foundational definition behind every "first population moment" used in §11 and the bridge into both MoM (uses $E(X^k)$) and MLE.

### Continuous Distributions

| Distribution | Probability Function | Mean | Variance | Typical Use in This Unit |
|---|---|---|---|---|
| **Uniform** | $f(y)=\dfrac1{\theta_2-\theta_1};\ \theta_1\le y\le\theta_2$ | $\dfrac{\theta_1+\theta_2}{2}$ | $\dfrac{(\theta_2-\theta_1)^2}{12}$ | Reference distribution |
| **Normal** | $f(y)=\dfrac1{\sigma\sqrt{2\pi}}\exp\!\left[-\dfrac1{2\sigma^2}(y-\mu)^2\right]$ | $\mu$ | $\sigma^2$ | §11 MoM for μ, σ² |
| **Exponential** | $f(y)=\dfrac1\beta e^{-y/\beta};\ \beta>0$ | $\beta$ | $\beta^2$ | §11 MoM examples, checkout exercise |
| **Gamma** | $f(y)=\dfrac1{\Gamma(\alpha)\beta^\alpha}y^{\alpha-1}e^{-y/\beta}$ | $\alpha\beta$ | $\alpha\beta^2$ | §11 MoM for α, β |
| **Chi-square** | $f(y)=\dfrac{y^{(v/2)-1}e^{-y/2}}{2^{v/2}\Gamma(v/2)}$ | $v$ | $2v$ | Reference (used later for variance inference) |
| **Beta** | $f(y)=\dfrac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}y^{\alpha-1}(1-y)^{\beta-1}$ | $\dfrac{\alpha}{\alpha+\beta}$ | $\dfrac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ | Reference |

### Discrete Distributions

| Distribution | Probability Function | Mean | Variance | Typical Use |
|---|---|---|---|---|
| **Binomial** | $p(y)=\binom{n}{y}p^y(1-p)^{n-y}$ | $np$ | $np(1-p)$ | §5–§8 proportion examples |
| **Poisson** | $p(y)=\dfrac{\lambda^y e^{-\lambda}}{y!}$ | $\lambda$ | $\lambda$ | Reference |

**Why It Matters.** These tables directly feed MoM: exponential row → $E(Y)=\beta$; gamma row → $E(Y)=\alpha\beta$. Each distribution's mean formula is exactly what gets equated to $\bar x$ (or a solved system, for 2+ parameters) in §11.

**AI/ML Connection.** The Bernoulli/Binomial and Normal distributions here are the same ones underlying logistic regression (Bernoulli-distributed labels) and linear regression (Normal-distributed noise) — the parameter-estimation machinery in this unit (MoM, and later MLE) is exactly what "fitting" such models means mathematically.

---

## Sampling Distributions — Summary Table

| Object | What it is | Randomness |
|---|---|---|
| Population distribution | $f(x)$, the model for one observation | Describes one $X_i$ |
| Sample | $X_1,\ldots,X_n$ | Each $X_i$ random before observed |
| Statistic / estimator | $T(X_1,\ldots,X_n)$, e.g. $\bar X$ | Random — has its own distribution |
| Sampling distribution | Distribution of $T(X_1,\ldots,X_n)$ across repeated samples | What §5–§8's mean/variance criteria are computed over |
| Estimate | $T(x_1,\ldots,x_n)$, a single number | Fixed, not random |

**Important Note.** This unit (Lectures 1–2) does not derive the exact sampling distribution of $\bar X$ or the CLT formally — it uses the *existence* of a sampling distribution (§3 above) to justify computing $E(\hat\Theta)$ and $V(\hat\Theta)$. Full derivations of sampling distributions (e.g. of $\bar X$ under Normal populations, χ² for $S^2$) belong to the next unit, Interval Estimation.

---

## Interval Estimation, Hypothesis Testing & Bayesian Inference — Orientation Only

**Important Note.** These three topics are **not covered** in Lectures 1–2 and therefore are not taught in depth here — doing so would introduce un-sourced material. They are included only as a brief orientation, because the lecture explicitly places Point Estimation as the first of three pillars (§ Statistical Foundations above), and because they help distinguish "judging/constructing an estimate" from these related-but-different inference goals.

- **Interval Estimation** (future lecture): instead of one number, gives a *range* of plausible values for θ, with a stated confidence level — will build directly on the sampling distributions and estimators developed here.
- **Hypothesis Testing** (future lecture): makes a binary accept/reject decision about a pre-stated claim about θ, as previewed in §1's floor-wax example — uses a test statistic, which is again a statistic in the same sense defined in this guide.
- **Bayesian Inference** (not part of this course's early units, but foundational for later probabilistic-ML material): treats θ itself as random, combining a prior $p(\theta)$ with the likelihood via Bayes' rule to get a posterior $p(\theta\mid x)$ — contrasted with the frequentist view used throughout this guide, where θ is fixed. For orientation only:

| Concept | Frequentist (this guide) | Bayesian |
|---|---|---|
| Parameter θ | Fixed, unknown constant | Modeled as a random variable |
| Point summary | MLE / MoM / unbiased estimator | Posterior mean or MAP |
| Interval | Confidence interval (future lecture) | Credible interval |

**[UNCLEAR IN SOURCE — DO NOT INFER]**: the lecture does not specify when Bayesian inference is introduced in this course; it is included above purely as conceptual orientation, not as examinable content from Lectures 1–2.

---

## Detailed Derivations (Consolidated)

1. **$E(\hat p)=p$ for the Bernoulli/Binomial proportion estimator** — see §5, Example (MoM estimator of *p* is unbiased).
2. **Bias–variance decomposition of MSE** — see §7, full four-step derivation.
3. **MoM estimators for exponential, normal, gamma, shifted-exponential** — see §11, each derived by equating population and sample moments and solving.

These three derivations are the ones explicitly flagged by the lecture as things students must be able to *reproduce*, not just recognize (see Exam Preparation below).

---

## Problem-Solving Procedures

### Estimation Problem Checklist
```text
1. Identify the parameter θ.
2. Identify the distribution family (statistical model).
3. Identify the sample X₁,…,Xₙ and its size n.
4. Choose or derive an estimator (e.g. via Method of Moments).
5. Determine its bias: compute E(estimator) and compare to θ.
6. Determine its variance if asked.
7. Determine its MSE if asked (Variance + Bias²).
8. Compare to alternative estimators via relative efficiency if asked.
9. Plug in the numeric data to get the estimate.
10. Interpret the result in context.
```

### Method-of-Moments Problem Checklist
(See §10 — reproduced here for the exam-prep flow.)
```text
1. Count parameters m.
2. Write E(X),…,E(X^m) in terms of the parameters.
3. Equate to sample moments (1/n)ΣXᵢᵏ.
4. Solve the system for the m parameters.
5. Report the estimator(s), then substitute numbers if given.
```

---

## Comparisons

| Concept A | Concept B | Why the distinction matters |
|---|---|---|
| **Parameter** | **Statistic** | A parameter is fixed and about the population; a statistic is computable from the sample and is random before observation. |
| **Estimator** | **Estimate** | Estimator = general formula/random variable; estimate = one realized number. |
| **Population distribution** | **Sampling distribution** | The former describes one observation; the latter describes a statistic across repeated sampling — never interchange them. |
| **Unbiasedness** | **Efficiency** | Unbiasedness concerns the *center* of the sampling distribution; efficiency concerns its *spread* — efficiency is only meaningful once you've narrowed to unbiased candidates. |
| **Variance** | **MSE** | Variance measures spread around the estimator's *own mean*; MSE measures spread around the *true parameter* — they coincide only when unbiased. |
| **Method of Moments** | **Maximum Likelihood** | MoM matches moments (often easier, closed-form); MLE maximizes likelihood (generally more efficient, sometimes needs numerical optimization). |
| **Frequentist inference** | **Bayesian inference** | Frequentist: θ fixed, randomness in sampling. Bayesian: θ random, updated via a prior — a different philosophy, previewed above but not examined in this unit. |

---

## AI / Machine Learning Connections

```text
Probability
      ↓
Statistical Models   (e.g. Bernoulli labels, Gaussian noise)
      ↓
Point Estimation      (this guide: unbiasedness, efficiency, MSE, MoM)
      ↓
Machine Learning       (parameter fitting = estimator construction)
      ↓
Loss-function minimization   (MSE loss ↔ MSE; cross-entropy ↔ likelihood, next unit)
      ↓
Bias–Variance Tradeoff in ML   (regularization, ensembling, model capacity)
```

- **Bias–variance decomposition (§7)** ↔ the ML bias–variance tradeoff explaining overfitting/underfitting.
- **Unbiasedness vs. deliberately biased estimators (§5, §7)** ↔ why regularization (L1/L2, weight decay) intentionally introduces bias to reduce variance and improve generalization.
- **Method of Moments (§10–§11)** ↔ simple moment-matching initializations still used for mixture models / GANs before more expensive optimization (e.g., MLE, gradient descent) takes over.
- **Efficiency / Cramér–Rao bound (§6)** ↔ the idea of *sample efficiency*: how much data an algorithm needs to reach a target precision.
- **Method of Maximum Likelihood (previewed in §9)** ↔ the statistical foundation of training via cross-entropy (classification) and MSE loss (regression under Gaussian noise) — developed fully once MLE itself is covered in the next lecture.

*(Per the source-fidelity rule, connections here are drawn only as far as the lecture's own content supports — deeper ML training mechanics, e.g. full backpropagation or neural-network loss derivations, are outside the scope of a Point-Estimation lecture and are not fabricated here.)*

---

## Common Mistakes

For each: what's wrong → why → correct view → how to avoid it.

1. **Confusing $\hat\theta$ (a number) with $\hat\Theta$ (a random variable).** Wrong: treating a single estimate as if it has its own "variance." Correct: only the estimator (formula), not a realized estimate, has a sampling distribution with mean/variance. Avoid by always asking "am I talking about the formula or a specific computed number?"
2. **Assuming the "best" unbiased estimator is unique.** Wrong: picking any unbiased estimator and stopping. Correct: mean, median, trimmed mean can all be unbiased for μ (§5) — you must then compare variances (efficiency) to break the tie.
3. **Forgetting the bias² term in MSE.** Wrong: writing MSE = Variance unconditionally. Correct: MSE = Variance + Bias², reducing to Variance only when bias = 0. Avoid by writing the full decomposition every time before simplifying.
4. **Misreading the relative-efficiency direction.** Wrong: concluding the denominator's estimator is better when the ratio is < 1. Correct: ratio < 1 means the **numerator's** estimator has the smaller MSE and wins. Avoid by explicitly rewriting the inequality as $\text{MSE}(\hat\Theta_1)<\text{MSE}(\hat\Theta_2)$ before concluding.
5. **In two-parameter MoM problems, forgetting $E(X^2)$.** Wrong: only equating the first moment when there are two unknown parameters. Correct: an *m*-parameter model needs *m* moment equations (first *m* population moments = first *m* sample moments). Avoid by counting unknown parameters before writing any equations.
6. **Confusing population distribution with sampling distribution.** Wrong: assuming $V(\bar X)=\sigma^2$. Correct: $V(\bar X)=\sigma^2/n$ — the sampling distribution of the mean is tighter than the population distribution (Textbook Expansion, §3). Avoid by always specifying *which* random variable's distribution you mean.
7. **Treating the MLE/MoM (biased, divide-by-*n*) variance estimator as unbiased.** Wrong: assuming $\hat\sigma^2=\frac1n\sum(x_i-\bar x)^2$ is unbiased for $\sigma^2$. Correct: it is slightly biased; the $n-1$ version is the unbiased one (§11, Normal example). Avoid by explicitly checking $E(\hat\sigma^2)$ rather than assuming.

---

## Exam Preparation

### What You Must Understand
- The estimator ($\hat\Theta$) vs. estimate ($\hat\theta$) distinction.
- Why unbiasedness alone cannot select a unique "best" estimator.
- Why MSE, not variance alone, is the right criterion when comparing biased and unbiased estimators together.
- The two separate skills this unit teaches: **judging** estimators (§5–§8) vs. **constructing** them (§9–§11).

### What You Must Be Able to Derive
- The bias–variance decomposition of MSE (§7) — from first principles.
- $E(\hat p)=p$ for the Bernoulli/binomial proportion estimator (§5).
- MoM estimators for exponential, normal, and gamma models (§11), by equating population and sample moments.

### What You Must Be Able to Calculate
- Point estimates $\bar x$, $\hat p$ from raw data.
- MSE and relative efficiency given two estimators' variances/biases.
- MoM estimators, symbolically and numerically, for exponential, normal, gamma, and shifted-exponential models.

### What You Must Be Able to Interpret
- What "unbiased" means about a sampling distribution's center, not about any single estimate.
- What a computed MSE or relative-efficiency ratio implies about which estimator to prefer.
- What it means that $\hat\sigma^2$ (MoM/MLE form) is biased — in what direction, and why it still matters practically.

### What You Must Be Able to Compare
- Parameter vs. statistic; estimator vs. estimate; unbiasedness vs. efficiency; variance vs. MSE; Method of Moments vs. Maximum Likelihood.

### Self-Assessment Questions

1. *(Conceptual)* Given data $\{12.8, 9.4, 8.7, 11.6, 13.1, 9.8, 14.1, 8.5, 12.1, 10.3\}$, compute the point estimate of μ.
2. *(Numerical)* For a binomial experiment with *x*=15 successes out of *n*=25 trials, compute $\hat p$.
3. *(Definitional)* State the definition of an unbiased estimator in symbols.
4. *(Derivation)* Write the bias–variance decomposition of MSE.
5. *(Interpretation)* If $\text{MSE}(\hat\Theta_1)=4$ and $\text{MSE}(\hat\Theta_2)=9$, compute the relative efficiency of $\hat\Theta_2$ to $\hat\Theta_1$ and state which is preferred.
6. *(Derivation)* Derive the MME of λ for an exponential distribution given $E(X)=1/\lambda$.
7. *(Derivation, method-selection)* For a random sample from $N(\mu,\sigma^2)$, derive both MMEs.
8. *(Method-selection)* For a Gamma($\alpha,\beta$) sample, write the two moment equations before solving.
9. *(Numerical)* For the calculator-brand data (20 students, 10 own TI), compute $\hat p$ for TI ownership.
10. *(Numerical)* For the checkout-arrival data (14 observations, sum = 85), compute the MME of θ for $f(x)=\frac1\theta e^{-x/\theta}$.

### Answer Key

1. $\bar x = 11.04$
2. $\hat p = 0.60$
3. $E(\hat\Theta) = \theta$
4. $\text{MSE}(\hat\Theta) = V(\hat\Theta) + (\text{Bias})^2$
5. ratio $=4/9\approx0.44<1$ → $\hat\Theta_1$ is preferred (more efficient).
6. $\hat\lambda = 1/\bar x$
7. $\hat\mu=\bar x,\ \hat\sigma^2=\frac1n\sum(x_i-\bar x)^2$
8. $\bar X=\alpha\beta$, $\frac1n\sum X_i^2=\alpha(\alpha+1)\beta^2$
9. $\hat p = 0.50$
10. $\hat\theta \approx 6.07$

---

## Formula Sheet

### Estimation

| Formula | Meaning | Symbols | Conditions |
|---|---|---|---|
| $\bar x = \dfrac1n\sum_{i=1}^n x_i$ | Sample mean | $x_i$: observed data, $n$: sample size | — |
| $s^2 = \dfrac1n\sum_{i=1}^n (x_i-\bar x)^2$ | Sample variance (moment form) | — | Biased for population $\sigma^2$ (divides by $n$, not $n-1$) |
| $\hat p = x/n$ | Sample proportion | $x$: successes, $n$: trials | Binomial setting |
| $E(\hat\Theta) = \theta$ | Unbiasedness | $\hat\Theta$: estimator, $\theta$: true parameter | Expectation over sampling distribution |
| $\sigma^2_{\hat\theta_1} < \sigma^2_{\hat\theta_2} \Rightarrow \hat\Theta_1$ more efficient | Efficiency comparison | — | Only valid when both estimators are unbiased |
| $\text{MSE}(\hat\Theta) = E[(\hat\Theta-\theta)^2]$ | Mean Squared Error | — | Applies to biased or unbiased estimators |
| $\text{MSE}(\hat\Theta) = V(\hat\Theta) + (\text{Bias})^2$ | Bias–variance decomposition | — | Always holds |
| $\dfrac{\text{MSE}(\hat\Theta_1)}{\text{MSE}(\hat\Theta_2)}$ | Relative efficiency | if $<1$: $\hat\Theta_1$ more efficient | — |

### Moments (Method of Moments)

| Formula | Meaning | Symbols | Conditions |
|---|---|---|---|
| $E(X^k)$ | *k*-th population moment | — | Must exist/be finite |
| $\dfrac1n\sum_{i=1}^n X_i^k$ | *k*-th sample moment | — | — |
| $\hat\lambda = 1/\bar x$ (or $\hat\beta=\bar x$) | MME — Exponential | $\lambda$ rate, $\beta$ mean parameterization | One-parameter case |
| $\hat\mu=\bar x,\ \hat\sigma^2=\frac1n\sum(x_i-\bar x)^2$ | MME — Normal | $\mu,\sigma^2$: population mean/variance | Two-parameter case |
| $\hat\alpha=\dfrac{\bar X^2}{\frac1n\sum X_i^2-\bar X^2},\ \hat\beta=\dfrac{\frac1n\sum X_i^2-\bar X^2}{\bar X}$ | MME — Gamma | $\alpha$: shape, $\beta$: scale | Two-parameter case |
| $\hat\eta = \bar x - 1$ | MME — Shifted Exponential EXP(1,η) | $\eta$: shift/location | One-parameter case, mean $=1+\eta$ |

### Asymptotic (Preview Only)

| Formula | Meaning | Conditions |
|---|---|---|
| $V(\bar X) = \sigma^2/n$ | Variance of the sample mean | i.i.d. sample; used to justify why larger $n$ tightens the sampling distribution |
| $V(\hat\Theta)\ge \dfrac1{n\,I(\theta)}$ | Cramér–Rao lower bound | Textbook expansion; full derivation not covered in this lecture |

### AI/ML Connections

| Statistical Concept | ML Analogue |
|---|---|
| $\text{MSE}=V+\text{Bias}^2$ | Bias–variance tradeoff (generalization error decomposition) |
| Deliberately biased estimator with lower variance | Regularization (L1/L2, weight decay) |
| Method of Moments | Moment-matching initialization (e.g. mixture models, GANs) |
| Maximum Likelihood (previewed) | $\arg\max L(\theta;x) \equiv \arg\min[-\log L(\theta;x)]$ — foundation of cross-entropy / MSE loss training |

---

## Concept Map

```text
STATISTICAL INFERENCE
│
├── ESTIMATION
│    ├── Point Estimation  ◄── this guide
│    │     ├── What is a "good" estimator?
│    │     │     ├── Unbiasedness:        E(Θ̂) = θ
│    │     │     ├── Efficiency (var.):   smaller var among unbiased ⇒ better
│    │     │     ├── Mean Squared Error:  MSE = Var + Bias²
│    │     │     └── Relative Efficiency: MSE(Θ̂₁)/MSE(Θ̂₂) < 1 ⇒ Θ̂₁ better
│    │     │
│    │     └── How do we GET an estimator?
│    │           ├── Method of Moments   (equate E(Xᵏ) to (1/n)ΣXᵢᵏ)
│    │           └── Maximum Likelihood  (next topic — maximizes likelihood)
│    │
│    └── Interval Estimation  (future lecture)
│
└── TESTING OF HYPOTHESES  (future lecture)
```

```text
Likelihood  (previewed only)
     ↓
MLE
     ↓
Statistical Estimation
     ↓
Machine Learning Loss Functions   (cross-entropy, MSE loss)
```

---

## Glossary

- **Population** — the entire group of items/individuals of interest.
- **Sample** — a random subset of the population, ideally selected at random.
- **Parameter (θ)** — a fixed, usually unknown numerical property of the population.
- **Statistic** — any computable function of sample data (contains no unknown parameters).
- **Estimator ($\hat\Theta$)** — a statistic used as a rule for estimating a parameter; a random variable before data is observed.
- **Estimate ($\hat\theta$)** — the realized numeric value of an estimator, computed from one observed sample.
- **Sampling distribution** — the probability distribution of a statistic across all possible samples of a given size.
- **Bias** — $E(\hat\Theta)-\theta$; the systematic average deviation of an estimator from the truth.
- **Unbiased estimator** — one with bias equal to 0, i.e. $E(\hat\Theta)=\theta$.
- **Efficiency** — a comparison of variances among unbiased estimators; smaller variance = more efficient.
- **Mean Squared Error (MSE)** — $E[(\hat\Theta-\theta)^2]$; combines variance and squared bias.
- **Relative efficiency** — ratio of two estimators' MSEs, used to compare them (biased or unbiased).
- **Method of Moments (MoM)** — construction technique equating population moments to sample moments, solved for the parameters.
- **Maximum Likelihood Estimation (MLE)** — construction technique maximizing the probability (likelihood) of the observed data (previewed, not derived, in this guide).
- **Moment** — $E(X^k)$ (population) or $\frac1n\sum X_i^k$ (sample), for $k=1,2,\ldots$
- **Consistency** *(mentioned, not derived here)* — the property that an estimator converges to θ as $n\to\infty$; related to but distinct from unbiasedness.
- **Frequentist inference** — the paradigm used throughout this guide, where θ is fixed and randomness lives in the sampling.
- **Bayesian inference** *(preview only)* — the paradigm where θ is treated as random and updated via a prior and Bayes' rule.

---

## Final Review

### The Big Picture
The essential thread of Lectures 1 & 2 is a two-part logical structure. First, **judging** estimators — given several candidate formulas for estimating θ, how do we rank them? Unbiasedness narrows the field; efficiency (smallest variance) breaks ties among unbiased candidates; MSE generalizes the comparison to *any* pair of estimators; relative efficiency turns MSE into a simple ratio-based decision rule. Second, **constructing** estimators — given only the population model, how do we get an estimator in the first place? The Method of Moments answers this by equating theoretical moments to their empirical counterparts and solving. Maximum Likelihood, covered next, answers the same question differently.

### Essential Concepts
Estimator vs. estimate; parameter vs. statistic; unbiasedness; efficiency; MSE and its bias–variance decomposition; relative efficiency; the Method of Moments recipe.

### Essential Formulas
$E(\hat\Theta)=\theta$; $\text{MSE}=V+\text{Bias}^2$; the MoM estimators for exponential, normal, gamma, and shifted-exponential (see Formula Sheet).

### Essential Procedures
The Estimation Problem checklist and the Method-of-Moments checklist above — every worked example in this guide (batteries, bumpers, calculators, module failure times, survival times, checkout arrivals) is an instance of one of these two procedures.

### Essential Distinctions
Population distribution vs. sampling distribution; variance vs. MSE; unbiasedness vs. efficiency; Method of Moments vs. Maximum Likelihood.

### AI Connections
The bias–variance decomposition of MSE is the same algebra behind the ML bias–variance tradeoff; deliberately biased-but-lower-variance estimators are the statistical justification for regularization; Method of Moments is the ancestor of simple moment-matching initializations; Maximum Likelihood (previewed) is the statistical foundation beneath cross-entropy and MSE loss functions used to train most supervised ML models.

---

*Compiled for DSAI 307 (Dr. Rasha Mandouh), Statistical Inference — Zewail City of Science, Technology and Innovation. Primary source: Lectures 01–02 slide content (100% preserved), expanded with textbook context (Casella & Berger, DeGroot & Schervish, Wasserman, Rice) and AI/ML connections (Murphy, Bishop, Hastie–Tibshirani–Friedman) for a Data Science & AI undergraduate audience. Topics outside Lectures 1–2 (full MLE derivation, confidence intervals, hypothesis testing, Bayesian inference) are previewed for orientation only and are not treated as examinable content from this source.*
