# DSAI 307 — Statistical Inference
## Master Study Guide: Point Estimation (Lectures 1 & 2)

**Instructor:** Dr. Rasha Mandouh · **Compiled from:** Lecture slides 01–02, cross-referenced with Casella & Berger, Wasserman, DeGroot & Schervish, Rice, and Murphy (*Probabilistic ML*).

> **Legend used throughout:** 🟦 From the Lecture · 🟩 Textbook Expansion · 🟨 Additional Explanation · 🤖 AI/ML Connection

---

## Table of Contents

1. [Course Overview](#1-course-overview)
2. [The Big Picture — Where Point Estimation Fits](#2-the-big-picture--where-point-estimation-fits)
3. [Statistical Inference: Estimation vs. Hypothesis Testing](#3-statistical-inference-estimation-vs-hypothesis-testing)
4. [Point Estimation — Foundations](#4-point-estimation--foundations)
5. [Worked Examples: Point Estimates](#5-worked-examples-point-estimates)
6. [Estimators as Random Variables (Sampling Distributions)](#6-estimators-as-random-variables-sampling-distributions)
7. [Parameters We Commonly Estimate](#7-parameters-we-commonly-estimate)
8. [Unbiasedness](#8-unbiasedness)
9. [Efficiency (Variance)](#9-efficiency-variance)
10. [Mean Squared Error (MSE)](#10-mean-squared-error-mse)
11. [Relative Efficiency](#11-relative-efficiency)
12. [Methods of Point Estimation — Overview](#12-methods-of-point-estimation--overview)
13. [Method of Moments — Theory](#13-method-of-moments--theory)
14. [Method of Moments — Fully Worked Examples](#14-method-of-moments--fully-worked-examples)
15. [Quick Reference: Common Distributions](#15-quick-reference-common-distributions)
16. [Comparison Tables](#16-comparison-tables)
17. [Exam Preparation](#17-exam-preparation)
18. [Master Formula Sheet](#18-master-formula-sheet)
19. [Concept Map](#19-concept-map)
20. [Glossary](#20-glossary)
21. [Final Review](#21-final-review)

---

## 1. Course Overview

**DSAI 307 — Statistical Inference**

### Course Contents
- Point Estimation
- Interval Estimation
- Testing of Hypotheses

### Grading Breakdown

| Component | Weight |
|---|---|
| Class Work | 40% |
| Midterm | 20% |
| Final Exam | 40% |

This guide covers **Point Estimation** — the first pillar of the course, and conceptually the foundation for everything that follows (confidence intervals are built *around* point estimates; hypothesis tests use estimators as test statistics).

---

## 2. The Big Picture — Where Point Estimation Fits

🟦 The lecture frames the whole subject with this pipeline:

```text
Probability
   ↓
Random Variables
   ↓
Distributions
   ↓
Sampling
   ↓
Sampling Distributions
   ↓
Statistical Inference
   ├── Estimation
   │     ├── Point Estimation      ◄── THIS GUIDE
   │     └── Interval Estimation
   └── Hypothesis Testing
```

🟨 **Intuition.** Everything upstream of "Statistical Inference" is *probability*: given a known model, what data will it produce? Everything from "Statistical Inference" downward is the reverse problem — *inference*: given observed data, what can we say about the unknown model that produced it? Point estimation is the simplest form of this reversal: collapse all the sample information into **one number** that stands in for an unknown population quantity.

🟩 **Textbook framing (Casella & Berger, Ch. 7).** This is the classical frequentist paradigm: the parameter θ is treated as a fixed but unknown constant, and randomness lives entirely in the sampling mechanism. This is in contrast to the Bayesian paradigm (mentioned briefly for comparison in §16), where θ itself is treated as a random variable with a prior distribution.

---

## 3. Statistical Inference: Estimation vs. Hypothesis Testing

### 3.1 Definition

🟦 > **Statistical inference** consists of those methods by which one makes inferences or generalizations about a population.

Statistical methods use the information in a **sample**, selected at random from the population, to draw conclusions about the **population** as a whole.

```text
POPULATION  ─────────sampling─────────►  SAMPLE
    ▲                                        │
    └───────────── inference ────────────────┘
```

### 3.2 The Two Major Branches

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

### 3.3 Distinguishing the Two — Worked Contrast

🟦 **Estimation example.** A candidate for public office samples 100 eligible voters to estimate the true proportion favoring him. The sample fraction estimates the unknown population proportion; knowing the sampling distribution of the proportion tells us how accurate that estimate is. → This is **estimation**: we want a number for an unknown quantity.

🟦 **Hypothesis-testing example.** We want to know whether floor wax brand A is more scuff-resistant than brand B. We state a hypothesis ("A is better than B"), test it, and **accept or reject** it. → This is **hypothesis testing**: we are not estimating a value, we are making a binary decision about a pre-stated claim.

🟨 **Why It Matters.** The distinction matters because the *question shapes the tool*: "What is it?" (estimation) needs an estimator and a measure of its precision; "Is it true?" (hypothesis testing) needs a decision rule with controlled error rates (Type I/II — covered in a later lecture). Both rely on the same underlying machine: **sampling distributions**.

---

## 4. Point Estimation — Foundations

### 4.1 Motivating Example

🟦 Let **μ** denote the true average breaking strength of wire connections used in bonding semiconductor wafers. A random sample of *n* = 10 connections is tested, giving observed strengths $x_1, \ldots, x_{10}$. The **sample mean** is used to draw a conclusion about μ.

### 4.2 Objective

🟦 > The objective of point estimation is to select a **single number**, based on sample data, that represents a sensible value for **θ** (a general parameter).

### 4.3 Formal Definitions

🟦 > **Point Estimator / Point Estimate.** A **point estimate** of a population parameter θ is a single numerical value $\hat\theta$ of a statistic $\hat\Theta$. The statistic $\hat\Theta$ is called the **point estimator**.

| Symbol | Name | What it is |
|---|---|---|
| $\hat\Theta$ (capital-hat) | **Estimator** | A rule / formula — a random variable, since it is a function of the (not-yet-observed) sample |
| $\hat\theta$ (lowercase-hat) | **Estimate** | A specific number, computed once real data is plugged in |

🟨 **Intuition.** Think of $\hat\Theta$ as a *recipe* (e.g., "add up the numbers and divide by how many there are") and $\hat\theta$ as the *dish* you get after actually cooking it with today's ingredients (today's sample). The recipe is fixed and reusable; the dish varies every time you cook.

**Key relationships:**
- $\bar x$ (a value of the statistic $\bar X$) → point estimate of the population mean **μ**
- $\hat p = x/n$ → point estimate of the true proportion **p** in a binomial experiment

🟩 **Textbook Expansion (Casella & Berger).** More generally, an estimator is *any* function $W(X_1,\ldots,X_n)$ of the sample that does not depend on unknown parameters (it must be computable from data alone — this is what makes it a *statistic*). Point estimation theory studies how to construct such functions and how to rank competing choices.

---

## 5. Worked Examples: Point Estimates

### Example 5.1 — Battery Lifetimes (Point Estimate of μ)

**Setup:** μ = true average lifetime of batteries. Sample of *n* = 3: $x_1=5.0,\ x_2=6.4,\ x_3=5.9$ (hours).

$$\bar x = \frac{5.0+6.4+5.9}{3} = 5.77$$

**Interpretation:** 5.77 hours is our single best guess for μ given this sample.

### Example 5.2 — Car Bumper Damage (Point Estimate of a Proportion *p*)

**Setup:** 25 controlled crashes at 10 mph; *X* = number resulting in no visible damage; *p* = true proportion with no damage. Observed *x* = 15.

$$\hat p = \frac{X}{n}\quad\text{(estimator)} \qquad \hat p = \frac{15}{25} = 0.60\quad\text{(estimate)}$$

### Example 5.3 — Sample Mean from Normal Data

**Setup:** *n* = 10 from a normal population: $12.8,\ 9.4,\ 8.7,\ 11.6,\ 13.1,\ 9.8,\ 14.1,\ 8.5,\ 12.1,\ 10.3$.

$$\bar x = \frac{110.4}{10} = 11.04$$

This dataset is reused throughout the guide (§8, §17.4) since it is also used to illustrate unbiasedness.

---

## 6. Estimators as Random Variables (Sampling Distributions)

🟦 Before data are collected, the observations are random variables $X_1,\ldots,X_n$. **Any function of them (any statistic) is therefore also a random variable.**

```text
Population
   │
   ├── Sample 1 of n=10  →  x̄ = 10.2
   ├── Sample 2 of n=10  →  x̄ = 10.4
   │        ⋮                  ⋮
```

Each new sample yields a (potentially) different value of $\bar x$. This is exactly why $\bar X$ and $S^2$ are called **statistics**: they are random variables with their own probability distribution — the **sampling distribution**.

🟨 **Why It Matters.** Since $\hat\Theta$ is a random variable, it has a mean $E(\hat\Theta)$ and a variance $V(\hat\Theta)$. These two numbers are exactly what let us judge whether one estimator is "better" than another (§8–§11) — this is the conceptual bridge into the rest of the guide.

🟩 **Textbook Expansion (Rice, Ch. 5).** For $\bar X$ from an i.i.d. sample with population mean μ and variance $\sigma^2$: $E(\bar X)=\mu$ and $V(\bar X)=\sigma^2/n$. As *n* grows, the sampling distribution of $\bar X$ concentrates more tightly around μ (Law of Large Numbers), and by the **Central Limit Theorem** it becomes approximately normal regardless of the shape of the population distribution — this is what later justifies confidence intervals built from $\bar X$.

---

## 7. Parameters We Commonly Estimate

🟦 We often need to estimate:

1. The mean **μ** of a single population
2. The variance **σ²** (or σ) of a single population
3. The proportion **p** of items in a class of interest
4. The difference in means of two populations, **μ₁ − μ₂**
5. The difference in two proportions, **p₁ − p₂**

### Reasonable Point Estimates

| Parameter | Point Estimate | Description |
|---|---|---|
| μ | $\hat\mu = \bar x$ | Sample mean |
| σ² | $\hat\sigma^2 = s^2$ | Sample variance |
| p | $\hat p = x/n$ | Sample proportion |
| μ₁ − μ₂ | $\bar x_1-\bar x_2$ | Difference of sample means (independent samples) |
| p₁ − p₂ | $\hat p_1-\hat p_2$ | Difference of sample proportions (independent samples) |

---

## 8. Unbiasedness

### 8.1 The Central Question

🟦 What properties make one estimator preferable to another? Multiple statistics can estimate the same parameter — for μ we could use the sample **mean** $\bar X$ or the sample **median** $\tilde X$. We need objective criteria.

### 8.2 Illustrating the Problem: Mean vs. Median

**Sample 1:** 2, 5, 11 (true μ = 4, known only for illustration)
- $\bar x = 6$, $\tilde x = 5$ → median (5) closer to 4 → median wins *this time*.

**Sample 2:** 2, 6, 7 (same population)
- $\bar x = 5$, $\tilde x = 6$ → mean (5) closer to 4 → mean wins *this time*.

🟨 **Conclusion.** We cannot judge an estimator case-by-case (we never know the true μ in practice). We must pick a rule **in advance**, based on a property that holds *on average across all possible samples* — this motivates unbiasedness.

### 8.3 Formal Definition

🟦 > Let $\hat\Theta$ be an estimator of θ. We would like the sampling distribution of $\hat\Theta$ to have a mean equal to the parameter it estimates. Such an estimator is **unbiased**:
> $$\mu_{\hat\Theta} = E(\hat\Theta) = \theta$$

🟨 **Intuition.** On average — across the (infinite) population of all possible samples of size *n* — the estimator neither systematically over- nor under-shoots the truth. Any single estimate can still be off; unbiasedness is a statement about the *long-run average*, not about any one sample.

**Bias** is formally defined as $\text{Bias}(\hat\Theta) = E(\hat\Theta) - \theta$; unbiased means Bias = 0.

### Example 8.1 — Multiple Unbiased Estimators of μ

Using the *n* = 10 dataset from Example 5.3:

| Statistic | Formula | Value |
|---|---|---|
| Sample mean | $\bar x$ | **11.04** |
| Sample median | $(10.3+11.6)/2$ | **10.95** |
| 10% trimmed mean | drop top/bottom 10%, average rest | **10.98** |

All three can be shown to be **unbiased estimators of μ** (for a symmetric population like the normal).

> 🟨 **Key takeaway.** Because the unbiased estimator is *not unique*, unbiasedness alone cannot crown a single "best" estimator — we need a second criterion: **variance/efficiency** (§9).

### Example 8.2 — MoM Estimator of *p* Is Unbiased (Bernoulli)

**Setup:** $X_1,\ldots,X_n$ i.i.d. Bernoulli(*p*). The moment estimator is $\hat p = \frac{1}{n}\sum X_i = Y/n$, where $Y=\sum X_i \sim \text{Binomial}(n,p)$.

**Derivation.** Since $E(Y) = np$ for a binomial random variable:
$$E(\hat p) = E\!\left(\frac{Y}{n}\right) = \frac{1}{n}E(Y) = \frac{1}{n}\cdot np = p \quad\checkmark$$

**Interpretation:** $\hat p = Y/n$ is unbiased for *p*.

🟩 **Limitations & Misconceptions.** Unbiasedness is a *population-level* guarantee, not a per-sample one — a single estimate can still be far from θ. Also, unbiased estimators are not always the most *useful*: an estimator can be unbiased yet have enormous variance, making individual estimates unreliable (this motivates §9–§10).

🤖 **AI/ML Connection.** In ML, we usually *don't* insist on unbiasedness — we optimize expected loss (bias² + variance, i.e., MSE-like objectives) directly, and often deliberately introduce bias (e.g., L2/L1 regularization, shrinkage, weight decay) to reduce variance and improve generalization. This is the statistical root of the **bias–variance tradeoff** used to explain overfitting/underfitting.

---

## 9. Efficiency (Variance)

### 9.1 Comparing Unbiased Estimators

🟦 > If $\hat\Theta_1$ and $\hat\Theta_2$ are both unbiased estimators of θ, we prefer the one whose sampling distribution has **smaller variance**.
> If $\sigma^2_{\hat\theta_1} < \sigma^2_{\hat\theta_2}$, then $\hat\Theta_1$ is **more efficient** than $\hat\Theta_2$.

🟨 **Intuition (Figure 1 in slides).** Picture two bell-shaped sampling distributions, both centered exactly at θ (both unbiased), but one is narrower/peakier than the other. The peakier one gives estimates that cluster tightly around the truth — you're less likely to be far off on any given sample. That is efficiency.

### 9.2 Three-Estimator Comparison

Consider $\hat\Theta_1, \hat\Theta_2, \hat\Theta_3$ estimating the same θ:
- $\hat\Theta_1, \hat\Theta_2$ are unbiased (centered at θ); $\hat\Theta_3$ is **biased** (centered elsewhere) and is disqualified on that basis alone.
- Between the two unbiased ones, $\hat\Theta_1$ has smaller variance → more efficient.
- **Overall best:** $\hat\Theta_1$ — unbiased *and* most efficient.

**Decision hierarchy so far:**
1. Prefer unbiased estimators.
2. Among unbiased estimators, prefer the smallest variance.

🟩 **Textbook Expansion — Cramér–Rao Lower Bound (Casella & Berger, Ch. 7.3).** Among all unbiased estimators, there is a theoretical floor on how small the variance can get, given by the Fisher Information $I(\theta)$:
$$V(\hat\Theta) \ge \frac{1}{n\, I(\theta)}$$
An unbiased estimator that *achieves* this bound is called an **efficient estimator** in the strict sense (not merely "more efficient than some alternative"). This is the rigorous foundation behind the informal "smaller variance is better" rule from the lecture.

🤖 **AI/ML Connection.** Efficiency parallels the notion of *sample efficiency* in ML: an estimator (or learning algorithm) that reaches a target accuracy with fewer samples is more "efficient," directly analogous to a lower-variance estimator needing less data to pin down θ precisely.

---

## 10. Mean Squared Error (MSE)

### 10.1 Why We Need It

🟦 Sometimes a **biased** estimator is unavoidable or even preferable (no unbiased estimator exists, or a biased one has much better overall behavior). We need one criterion that captures **both** bias and variance at once — the **MSE**.

🟨 **Intuition (Figure 3 in slides).** A biased $\hat\Theta_1$ can have a much smaller variance than an unbiased $\hat\Theta_2$, even though $\hat\Theta_1$ isn't centered exactly at θ. If the variance savings outweigh the bias penalty, the biased-but-tight estimator can actually be the better overall choice.

### 10.2 Formal Definition

🟦 > $$\text{MSE}(\hat\Theta) = E\big[(\hat\Theta - \theta)^2\big]$$

### 10.3 The Bias–Variance Decomposition (key derivation — memorize)

**Derivation.** Insert and subtract $E(\hat\Theta)$:
$$\text{MSE}(\hat\Theta) = E\big[(\hat\Theta - E(\hat\Theta)) + (E(\hat\Theta)-\theta)\big]^2$$
Expanding the square, the cross term vanishes in expectation (since $E\big[\hat\Theta - E(\hat\Theta)\big]=0$), leaving:
$$\text{MSE}(\hat\Theta) = \underbrace{E\big[\hat\Theta - E(\hat\Theta)\big]^2}_{V(\hat\Theta)} + \underbrace{\big[\theta - E(\hat\Theta)\big]^2}_{(\text{bias})^2}$$
$$\boxed{\text{MSE}(\hat\Theta) = V(\hat\Theta) + (\text{Bias})^2}$$

**Special case:** if $\hat\Theta$ is unbiased, Bias = 0, so $\text{MSE}(\hat\Theta) = V(\hat\Theta)$ — consistent with §9, where unbiased estimators were compared by variance alone.

🟩 **Textbook Expansion.** This decomposition is one of the most reused identities in all of statistics and ML: it appears verbatim (with different notation) in supervised learning theory, where the expected test error of a prediction function decomposes into (squared) bias, variance, and irreducible noise.

🤖 **AI/ML Connection.** This is *literally* the same algebra behind the ML **bias–variance tradeoff**: a high-capacity model (e.g., a deep, unregularized network) tends to have low bias but high variance (overfits); a low-capacity model (e.g., linear regression on nonlinear data) has high bias but low variance (underfits). Regularization, ensembling, and early stopping are all tools for moving along this same tradeoff curve to minimize an MSE-like generalization error.

🟨 **Common misconception.** Students often write "MSE = Variance" and forget the bias² term — this is only true in the unbiased special case.

---

## 11. Relative Efficiency

### 11.1 Definition

🟦 For two estimators $\hat\Theta_1,\hat\Theta_2$ of the same θ (biased or unbiased):
$$\text{Relative efficiency of } \hat\Theta_2 \text{ to } \hat\Theta_1 = \frac{\text{MSE}(\hat\Theta_1)}{\text{MSE}(\hat\Theta_2)}$$

### 11.2 Interpretation Rule

🟦 > If this ratio is **less than 1**, $\hat\Theta_1$ is the more efficient estimator (smaller MSE).

**Reading it correctly:**
$$\frac{\text{MSE}(\hat\Theta_1)}{\text{MSE}(\hat\Theta_2)} < 1 \iff \text{MSE}(\hat\Theta_1) < \text{MSE}(\hat\Theta_2) \implies \hat\Theta_1 \text{ wins}$$

🟨 **Common pitfall.** It is very easy to reverse this direction under exam pressure — always translate the ratio back into "which MSE is smaller" before concluding.

### Exercise 11.1 — Calculator Brands (Proportion Estimation)

**Setup:** 20 students; brands owned: `T T H T C T T S C H / S S T H C T T T H T` (T = Texas Instruments).

**Solution.** Count T's: Row 1 → 5, Row 2 → 5, total = 10 out of *n* = 20.
$$\hat p = \frac{10}{20} = 0.50$$

**Interpretation:** an estimated 50% of such students own a TI calculator.

---

## 12. Methods of Point Estimation — Overview

### 12.1 Why We Need "Methods"

🟦 Unbiasedness, efficiency, and MSE (§8–§11) tell us how to **judge** estimators — they say nothing about how to **construct** one in the first place. Two systematic construction methods:

1. **Method of Moments (MoM)** — equate population moments to sample moments.
2. **Maximum Likelihood Estimation (MLE)** — maximize the likelihood of the observed data (starts at the end of this unit / beginning of the next lecture).

### 12.2 Comparison of the Two Methods

| Aspect | Method of Moments | Maximum Likelihood |
|---|---|---|
| General efficiency | Generally *less* efficient | Generally **preferable** |
| Ease of computation | Often easier, closed-form | Can require solving harder optimization/likelihood equations |
| Unbiasedness | Can be unbiased | Can also be unbiased |

🟨 Both **can** yield unbiased estimators; MLE is generally favored for efficiency (it often attains or approaches the Cramér–Rao bound asymptotically), while MoM is valued for computational simplicity and as a good **starting guess** for iterative MLE optimization.

🤖 **AI/ML Connection.** MLE is the direct ancestor of how most ML models are trained: minimizing cross-entropy loss for classification, or mean-squared error for regression under a Gaussian-noise assumption, are both special cases of maximizing likelihood. Method of Moments corresponds to simpler "moment-matching" techniques still used for quick parameter initialization (e.g., initializing mixture-model or GAN training).

---

## 13. Method of Moments — Theory

### 13.1 General Idea

🟦 > Equate **population moments** (expected values) to the corresponding **sample moments**. Population moments are functions of the unknown parameters; solving the resulting equations yields the estimators.

### 13.2 Formal Definition of Moments

🟦 > Let $X_1,\ldots,X_n$ be a random sample from $f(x)$ (discrete pmf or continuous pdf).
> - **k-th population moment:** $E(X^k)$, $k=1,2,\ldots$
> - **k-th sample moment:** $\dfrac{1}{n}\sum_{i=1}^n X_i^k$, $k=1,2,\ldots$

### 13.3 First Moment — the Simplest Case

- First population moment: $E(X) = \mu$
- First sample moment: $\hat\mu = \frac{1}{n}\sum X_i = \bar X$
- Equate: $\hat\mu = \bar X$

> **The sample mean is the moment estimator of the population mean** — the fastest shortcut for any one-parameter distribution whose mean directly *is* that parameter (exponential mean = β, gamma mean = αβ, shifted-exponential mean = $1+\eta$, etc.).

### 13.4 General Case (*m* Parameters)

🟦 > Let $\theta_1,\ldots,\theta_m$ be *m* unknown parameters. The **moment estimators** $\hat\Theta_1,\ldots,\hat\Theta_m$ are found by equating the first *m* population moments to the first *m* sample moments and solving simultaneously.

**Practical recipe:**
1. Count the unknown parameters *m*.
2. Write $E(X), E(X^2), \ldots, E(X^m)$ as functions of the parameters.
3. Set each equal to $\frac1n\sum X_i^k$.
4. Solve the *m×m* system for the MMEs.

🟩 **Assumptions.** MoM implicitly assumes the relevant population moments exist and are finite (e.g., it fails for heavy-tailed distributions like the Cauchy, where $E(X)$ doesn't exist), and that the sample is i.i.d. from the assumed family.

---

## 14. Method of Moments — Fully Worked Examples

### Example 14.1 — Exponential Distribution (One Parameter)

**Setup:** $X_1,\ldots,X_n \sim \text{Exponential}(\lambda)$.

$$E(X) = \frac{1}{\lambda} = \bar x \implies \boxed{\hat\lambda = \frac{1}{\bar x}}$$

### Example 14.2 — Numerical: Electronic Module Failure Times

**Data (hrs), n = 8:** 11.96, 5.03, 67.40, 16.07, 31.50, 7.73, 11.10, 22.38

$$\bar x = \frac{173.17}{8} = 21.65 \qquad \hat\lambda = \frac{1}{21.65} = 0.0462$$

### Example 14.3 — Normal Distribution (Two Parameters: μ, σ²)

**Step 1 — Population moments:** $E(X)=\mu$, $E(X^2)=\mu^2+\sigma^2$.

**Step 2 — Equate:**
$$\mu = \bar x \qquad \mu^2+\sigma^2 = \frac1n\sum x_i^2$$

**Step 3 — Solve:**
$$\hat\mu = \bar x \qquad \hat\sigma^2 = \frac{\sum(x_i-\bar x)^2}{n}$$

🟨 **Discussion (posed in lecture): is $\hat\mu$ unbiased?** Yes — $E(\bar X)=\mu$ always (§8). Note $\hat\sigma^2$ here divides by *n*, not *n−1*; this is the MLE/MoM form and is in fact **biased** for $\sigma^2$ (it slightly underestimates on average). The unbiased version uses $n-1$ in the denominator — this contrast is a classic exam trap.

### Example 14.4 — Gamma Distribution (Two Parameters: α, β)

**Step 1 — Population moments:** $E(X)=\alpha\beta$, $E(X^2)=\alpha(\alpha+1)\beta^2$.

**Step 2 — Equate & solve:**
$$\bar X = \alpha\beta \qquad \frac1n\sum X_i^2 = \alpha(\alpha+1)\beta^2$$
$$\hat\alpha = \frac{\bar X^2}{\frac1n\sum X_i^2 - \bar X^2} \qquad \hat\beta = \frac{\frac1n\sum X_i^2 - \bar X^2}{\bar X}$$

**Numerical application (n = 20 survival times):** given $\bar x = 113.5$, $\frac1{20}\sum x_i^2 = 14{,}087.8$:
$$\hat\alpha = \frac{(113.5)^2}{14{,}087.8-(113.5)^2} = 10.7 \qquad \hat\beta = \frac{14{,}087.8-(113.5)^2}{113.5} = 10.6$$

### Example 14.5 — Two-Parameter (Shifted) Exponential

**Setup:** $X_i \sim \text{EXP}(1,\eta)$, mean $= 1+\eta$.

$$\bar x = 1+\hat\eta \implies \boxed{\hat\eta = \bar x - 1}$$

### Example 14.6 — Unbiasedness of the MoM Estimator for β (Exponential)

Since the MoM estimator is $\bar X$ and $E(X)=\beta$: $E(\bar X)=\beta$ ⟹ **unbiased.** ✓

### Exercise 14.7 — Supermarket Checkout Arrivals (Bridge to MLE)

**Data (min), n = 14:** 1, 2, 3, 7, 11, 4, 13, 12, 7, 3, 2, 11, 7, 2 → sum = 85.

**Model:** $f(x) = \frac1\theta e^{-x/\theta}$ (exponential, mean θ).

$$\bar x = \frac{85}{14}\approx 6.071 \qquad \boxed{\hat\theta \approx 6.07 \text{ minutes}}$$

🟨 This exercise is the slide's explicit bridge into **Maximum Likelihood Estimation**, the next topic (which reviews $E(X)$ definitions and the distribution table in §15 before deriving likelihood equations).

---

## 15. Quick Reference: Common Distributions

### 15.0 General Definition of the Mean

🟦 > For a random variable *X* with distribution $f(x)$:
> $$\mu = E(X) = \sum_x x\,f(x) \ \ (\text{discrete}) \qquad \mu = E(X) = \int_{-\infty}^{\infty} x\,f(x)\,dx \ \ (\text{continuous})$$

This is the foundational definition behind every "first population moment" used throughout §14 (e.g., $E(X)=1/\lambda$ for exponential, $E(X)=\alpha\beta$ for gamma), and it is the natural bridge into both MoM (uses $E(X^k)$) and MLE.

### 15.1 Continuous Distributions

| Distribution | Probability Function | Mean | Variance |
|---|---|---|---|
| **Uniform** | $f(y)=\dfrac{1}{\theta_2-\theta_1};\ \theta_1\le y\le\theta_2$ | $\dfrac{\theta_1+\theta_2}{2}$ | $\dfrac{(\theta_2-\theta_1)^2}{12}$ |
| **Normal** | $f(y)=\dfrac{1}{\sigma\sqrt{2\pi}}\exp\!\left[-\dfrac{1}{2\sigma^2}(y-\mu)^2\right]$ | $\mu$ | $\sigma^2$ |
| **Exponential** | $f(y)=\dfrac1\beta e^{-y/\beta};\ \beta>0$ | $\beta$ | $\beta^2$ |
| **Gamma** | $f(y)=\dfrac{1}{\Gamma(\alpha)\beta^\alpha}y^{\alpha-1}e^{-y/\beta}$ | $\alpha\beta$ | $\alpha\beta^2$ |
| **Chi-square** | $f(y)=\dfrac{y^{(v/2)-1}e^{-y/2}}{2^{v/2}\Gamma(v/2)}$ | $v$ | $2v$ |
| **Beta** | $f(y)=\dfrac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}y^{\alpha-1}(1-y)^{\beta-1}$ | $\dfrac{\alpha}{\alpha+\beta}$ | $\dfrac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |

### 15.2 Discrete Distributions

| Distribution | Probability Function | Mean | Variance |
|---|---|---|---|
| **Binomial** | $p(y)=\binom{n}{y}p^y(1-p)^{n-y}$ | $np$ | $np(1-p)$ |
| **Poisson** | $p(y)=\dfrac{\lambda^y e^{-\lambda}}{y!}$ | $\lambda$ | $\lambda$ |

> 🟨 These tables directly feed MoM: exponential row → $E(Y)=\beta$ (Examples 14.1/14.6); gamma row → $E(Y)=\alpha\beta$ (Example 14.4).

---

## 16. Comparison Tables

| Concept | Difference |
|---|---|
| **Parameter vs. Statistic** | A parameter (θ, μ, p) is a fixed but unknown number describing the *population*; a statistic ($\bar X$, $S^2$) is a computable function of the *sample*, hence a random variable. |
| **Estimator vs. Estimate** | Estimator $\hat\Theta$ is the general formula/random variable; estimate $\hat\theta$ is the specific number from one realized sample. |
| **MoM vs. MLE** | MoM matches moments (often easier, closed-form); MLE maximizes likelihood (generally more efficient, sometimes requires numerical optimization). |
| **Unbiasedness vs. Efficiency** | Unbiasedness concerns the *center* of the sampling distribution ($E(\hat\Theta)=\theta$); efficiency concerns its *spread* (variance) — a criterion only meaningful once you've narrowed to unbiased candidates. |
| **Variance vs. MSE** | Variance measures spread around the estimator's *own mean*; MSE measures spread around the *true parameter* — they coincide only when the estimator is unbiased. |
| **Frequentist vs. Bayesian** (preview) | Frequentist: θ is a fixed constant, randomness is in the sampling; inference uses sampling distributions of estimators. Bayesian: θ is itself a random variable with a prior distribution, updated to a posterior via observed data — not covered in Lectures 1–2 but relevant later in the course and heavily used in probabilistic ML. |

---

## 17. Exam Preparation

### 17.1 What You Must Understand
- The distinction between estimator ($\hat\Theta$) and estimate ($\hat\theta$).
- Why unbiasedness alone cannot select a unique "best" estimator.
- Why MSE, not variance, is the right criterion when comparing biased and unbiased estimators together.
- The logical structure: judge estimators (§8–§11) is a *separate skill* from construct estimators (§12–§14).

### 17.2 What You Must Be Able to Derive
- The bias–variance decomposition of MSE (§10.3) — from first principles, by inserting/subtracting $E(\hat\Theta)$.
- $E(\hat p) = p$ for the Bernoulli/binomial proportion estimator (§8, Example 8.2).
- MoM estimators for one- and two-parameter distributions (exponential, normal, gamma) by equating population and sample moments.

### 17.3 What You Must Be Able to Calculate
- Point estimates $\bar x$, $\hat p$ from raw data.
- MSE and relative efficiency given two estimators' variances/biases.
- MoM estimators, both symbolically and numerically, for exponential, normal, gamma, and shifted-exponential models.

### 17.4 Common Exam Traps
- **Confusing $\hat\theta$ (a number) with $\hat\Theta$ (a random variable)** — only the latter has a mean/variance.
- **Assuming the "best" unbiased estimator is unique** — mean, median, trimmed mean can all be unbiased for μ (Example 8.1); you need variance/efficiency to break the tie.
- **Forgetting the bias² term** and writing MSE = Variance when bias ≠ 0.
- **Misreading the relative-efficiency direction** — ratio < 1 means the *numerator's* estimator is better.
- **In two-parameter MoM problems, forgetting $E(X^2)$** — one-parameter problems only need the first moment; two-parameter problems need the first *and* second.

### 17.5 Self-Assessment Questions & Answer Key

1. *(Conceptual)* Given data $\{12.8, 9.4, 8.7, 11.6, 13.1, 9.8, 14.1, 8.5, 12.1, 10.3\}$, compute the point estimate of μ.
   **A:** $\bar x = 11.04$
2. *(Numerical)* For a binomial experiment with *x* = 15 successes out of *n* = 25 trials, compute $\hat p$.
   **A:** $\hat p = 0.60$
3. *(Definitional)* State the definition of an unbiased estimator in symbols.
   **A:** $E(\hat\Theta) = \theta$
4. *(Derivation)* Write the bias–variance decomposition of MSE.
   **A:** $\text{MSE}(\hat\Theta) = V(\hat\Theta) + (\text{Bias})^2$
5. *(Interpretation)* If $\text{MSE}(\hat\Theta_1)=4$ and $\text{MSE}(\hat\Theta_2)=9$, compute the relative efficiency of $\hat\Theta_2$ to $\hat\Theta_1$ and state which is preferred.
   **A:** ratio $=4/9\approx0.44<1$ → $\hat\Theta_1$ is preferred (more efficient).
6. *(Derivation)* Derive the MME of λ for an exponential distribution given $E(X)=1/\lambda$.
   **A:** $\hat\lambda = 1/\bar x$
7. *(Derivation, method-selection)* For a random sample from $N(\mu,\sigma^2)$, derive both MMEs.
   **A:** $\hat\mu=\bar x,\ \hat\sigma^2=\frac1n\sum(x_i-\bar x)^2$
8. *(Method-selection)* For a Gamma($\alpha,\beta$) sample, write the two moment equations before solving.
   **A:** $\bar X=\alpha\beta$, $\frac1n\sum X_i^2=\alpha(\alpha+1)\beta^2$
9. *(Numerical)* For the calculator-brand data (20 students, 10 own TI), compute $\hat p$ for TI ownership.
   **A:** $\hat p = 0.50$
10. *(Numerical)* For the checkout-arrival data (14 observations, sum = 85), compute the MME of θ for $f(x)=\frac1\theta e^{-x/\theta}$.
    **A:** $\hat\theta \approx 6.07$

---

## 18. Master Formula Sheet

| Concept | Formula |
|---|---|
| Sample mean | $\bar x = \dfrac1n\sum_{i=1}^n x_i$ |
| Sample variance (moment form) | $s^2 = \dfrac1n\sum_{i=1}^n (x_i-\bar x)^2$ |
| Sample proportion | $\hat p = x/n$ |
| *k*-th population moment | $E(X^k)$ |
| *k*-th sample moment | $\dfrac1n\sum_{i=1}^n X_i^k$ |
| Unbiasedness | $E(\hat\Theta) = \theta$ |
| More efficient (unbiased case) | $\sigma^2_{\hat\theta_1} < \sigma^2_{\hat\theta_2} \Rightarrow \hat\Theta_1$ more efficient |
| Mean Squared Error | $\text{MSE}(\hat\Theta) = E(\hat\Theta-\theta)^2$ |
| Bias–Variance decomposition | $\text{MSE}(\hat\Theta) = V(\hat\Theta) + (\text{Bias})^2$ |
| Relative efficiency | $\dfrac{\text{MSE}(\hat\Theta_1)}{\text{MSE}(\hat\Theta_2)}$; if $<1$, $\hat\Theta_1$ more efficient |
| MME — exponential | $\hat\lambda = 1/\bar x$ (or $\hat\beta=\bar x$, mean-β form) |
| MME — normal | $\hat\mu=\bar x,\ \ \hat\sigma^2=\dfrac1n\sum(x_i-\bar x)^2$ |
| MME — gamma | $\hat\alpha=\dfrac{\bar X^2}{\frac1n\sum X_i^2-\bar X^2},\ \ \hat\beta=\dfrac{\frac1n\sum X_i^2-\bar X^2}{\bar X}$ |
| MME — shifted exponential EXP(1,η) | $\hat\eta = \bar x - 1$ |
| Cramér–Rao lower bound | $V(\hat\Theta)\ge \dfrac{1}{n\,I(\theta)}$ |

---

## 19. Concept Map

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

---

## 20. Glossary

- **Population** — the entire group of items/individuals of interest.
- **Sample** — a subset of the population, ideally selected at random.
- **Parameter (θ)** — a fixed, usually unknown, numerical property of the population.
- **Statistic** — any computable function of sample data (contains no unknown parameters).
- **Estimator ($\hat\Theta$)** — a statistic used as a rule for estimating a parameter; a random variable before data is observed.
- **Estimate ($\hat\theta$)** — the realized numeric value of an estimator, computed from one observed sample.
- **Sampling distribution** — the probability distribution of a statistic across all possible samples of a given size.
- **Bias** — $E(\hat\Theta)-\theta$; the systematic average deviation of an estimator from the truth.
- **Unbiased estimator** — one with bias equal to 0, i.e. $E(\hat\Theta)=\theta$.
- **Efficiency** — a comparison of variances among unbiased estimators; the one with smaller variance is more efficient.
- **Mean Squared Error (MSE)** — $E[(\hat\Theta-\theta)^2]$; combines variance and squared bias.
- **Relative efficiency** — ratio of two estimators' MSEs, used to compare them (biased or unbiased).
- **Method of Moments (MoM)** — a construction technique that equates population moments to sample moments and solves for the parameters.
- **Maximum Likelihood Estimation (MLE)** — a construction technique that chooses parameter values maximizing the probability (likelihood) of the observed data.
- **Moment** — $E(X^k)$ (population) or $\frac1n\sum X_i^k$ (sample), for $k=1,2,\ldots$

---

## 21. Final Review

The essential thread of Lectures 1 & 2 is a **two-part logical structure**:

1. **Judging** estimators — given several candidate formulas for estimating θ, how do we rank them? Unbiasedness ($E(\hat\Theta)=\theta$) narrows the field; efficiency (smallest variance) breaks ties among unbiased candidates; MSE (= Variance + Bias²) generalizes the comparison to *any* pair of estimators, biased or not; relative efficiency turns MSE into a simple ratio-based decision rule.

2. **Constructing** estimators — given only the population model, how do we *get* an estimator in the first place? The Method of Moments answers this by equating theoretical moments $E(X^k)$ to their empirical (sample) counterparts and solving. Maximum Likelihood, covered next, answers the same question differently — by maximizing the probability of having observed the data you actually got.

Every worked example in this guide (batteries, bumpers, calculators, module failure times, survival times, checkout arrivals) is really just an instance of one of these two skills. Master the two skills, and every homework/exam problem in this unit reduces to "which skill does this question test, and which formula from §18 applies?"

🤖 Looking ahead: this entire estimation framework — unbiasedness, variance, MSE, moment matching, likelihood maximization — reappears throughout machine learning under different names (bias–variance tradeoff, loss minimization, MLE-based training objectives like cross-entropy and MSE loss). Mastering it here pays off directly in later ML coursework.

---

*Compiled for DSAI 307 (Dr. Rasha Mandouh), Statistical Inference — Zewail City of Science, Technology and Innovation. Primary source: Lectures 01–02 slide content (100% preserved), expanded with textbook context (Casella & Berger, Rice, DeGroot & Schervish) and AI/ML connections for a Data Science & AI undergraduate audience.*
