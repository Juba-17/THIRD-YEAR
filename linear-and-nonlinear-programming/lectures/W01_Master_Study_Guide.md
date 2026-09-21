# Math 303 — Linear and Nonlinear Programming for Computational Sciences
## Master Study Guide — Week 1: Fundamentals of Optimization & Classical Optimization Techniques

---

## Overview

This chapter builds the mathematical and conceptual foundation for the entire course. It answers three questions:

1. What is optimization, and why does it matter for engineering, science, and Artificial Intelligence?
2. How do we formally state an optimization problem?
3. How do we recognize an optimal point mathematically — for a function of one variable, and for a function of several variables?

The lecture does two things. First, it introduces optimization as a discipline (Fundamentals): what it is, how it is classified, how a study is carried out in practice, and how the general optimization problem is formulated. Second, it supplies the classical calculus-based tools needed to recognize and classify optima: the derivative/gradient conditions, the Hessian test, and the distinction between local and global extrema.

This guide teaches every statement, theorem, proof, example, and classification in the source lecture in full, and — where genuinely necessary — adds explanatory material, worked derivations, and commentary that the lecture leaves implicit.

---

## Learning Objectives

By the end of this chapter you should be able to:

1. Define optimization (mathematical programming) and explain its role in Operations Research.
2. Reproduce Table 1.1 (methods of Operations Research) and place any given technique into the correct category.
3. Carry out the five phases of an OR/MP study on a new problem.
4. Write the general constrained optimization problem in standard form, identify decision variables, objective function, equality/inequality constraints, and the feasible set.
5. Distinguish active vs. inactive inequality constraints, and explain why equality constraints are always active.
6. State and prove the necessary condition for a local extremum of a single-variable function, and explain its limitations.
7. State and prove the sufficient condition for a local extremum of a single-variable function using higher-order derivatives, and apply it to classify stationary points.
8. State and prove the necessary condition ($\nabla f(\mathbf{x}^*) = 0$) and the sufficient condition (Hessian positive definite) for a local minimum of a multivariable function.
9. Determine whether a symmetric matrix is positive definite, positive semi-definite, or indefinite, and connect this to local minima, local maxima, and saddle points.
10. Classify an optimization problem into its correct family: Linear Programming, Unconstrained Optimization, Convex Optimization, Quadratic Optimization, Geometric Programming, Integer Programming, Stochastic Programming, Multi-objective Optimization, or Modern/Heuristic Methods.
11. Connect the ideas above to AI/ML examples: neural network training, SVM margin maximization, and the bias–variance/generalization trade-off in regression.

---

## Prerequisites

You need the following background to follow this chapter.

- Single-variable calculus: derivatives, higher-order derivatives, Taylor's theorem with remainder.
- Multivariable calculus: partial derivatives, the gradient vector, the Hessian matrix, multivariable Taylor expansion.
- Linear algebra: vectors, matrices, matrix–vector and matrix–matrix multiplication, eigenvalues, quadratic forms, determinants, and principal minors.
- Basic real analysis vocabulary: limit, neighborhood, interval, domain of a function.
- Set notation: $\in$, $\subset$, $\{\cdot \mid \cdot\}$.

---

# Main Concepts

## Concept 1 — What Is Optimization?

### Definition

> Optimization (Programming) is the science that is concerned with the efficient use (allocation) of limited resources to meet certain objectives and satisfy predefined constraints.

### Intuition

Every optimization problem has the same skeleton: you have choices to make (variables), limited resources or rules to respect (constraints), and a single yardstick for “better” (the objective function).

### Why It Exists

Without a formal optimization framework, engineers and scientists would compare candidate designs one by one, by trial and error, with no guarantee that a better design is found.

### Table 1.1 — Methods of Operations Research

| Mathematical programming or optimization techniques | Stochastic process techniques | Statistical methods |
|---|---|---|
| Calculus methods | Statistical decision theory | Regression analysis |
| Calculus of variations | Markov processes | Cluster analysis, pattern recognition |
| Nonlinear programming | Queueing theory | Design of experiments |
| Geometric programming | Renewal theory | Discriminate analysis (factor analysis) |
| Quadratic programming | Simulation methods | |
| Linear programming | Reliability theory | |
| Dynamic programming | | |
| Integer programming | | |
| Stochastic programming | | |
| Separable programming | | |
| Multiobjective programming | | |
| Network methods: CPM and PERT | | |
| Game theory | | |

Modern or nontraditional optimization techniques include:
- Genetic algorithms
- Simulated annealing
- Ant colony optimization
- Particle swarm optimization
- Neural networks
- Fuzzy optimization

### AI / Machine Learning Connection

Nearly every training algorithm in machine learning is an instance of calculus-based optimization or nonlinear programming: training a neural network is a large-scale nonlinear optimization problem, and model selection is also fundamentally optimization.

---

## Concept 2 — Motivating Examples of Optimization

### Example 1 — Minimizing the Cost of Business-Trip Airfare

A business trip between Cairo and Dubai repeats for 5 weeks. A regular round trip costs $\$400$, but a 20% discount is granted when the trip spans a weekend.

The objective is to minimize total travel cost, subject to the scheduling restriction.

The relevant alternatives are:

1. 5 regular round trips.
2. A mixture of one-way and discounted round trips.
3. 5 discounted round trips.

Cost calculations:

$$
\text{Alternative 1 cost} = 5 \times 400 = \$2000
$$

$$
\text{Alternative 2 cost} = 0.75\times 400 + 4\times(0.8\times 400) + 0.75\times 400 = \$1880
$$

$$
\text{Alternative 3 cost} = 5 \times (0.8\times 400) = \$1600
$$

Hence Alternative 3 is the best choice.

### Example 2 — Maximum-Area Rectangle from a Fixed-Length Wire

Let the rectangle have width $w$ and height $h$, and let the wire length be $L$.

The problem is:

$$
\max_{w,h} \; wh
$$

subject to

$$
L = 2(w+h), \quad w>0, \quad h>0.
$$

Using the constraint,

$$
h = \frac{L}{2} - w.
$$

Substitute into the area:

$$
A(w) = w\left(\frac{L}{2} - w\right) = \frac{L}{2}w - w^2.
$$

Differentiate:

$$
A'(w) = \frac{L}{2} - 2w.
$$

Set the derivative to zero:

$$
\frac{L}{2} - 2w = 0
\implies w = \frac{L}{4}.
$$

Then,

$$
h = \frac{L}{2} - \frac{L}{4} = \frac{L}{4}.
$$

Therefore,

$$
w = h = \frac{L}{4}.
$$

So the maximum-area rectangle is a square.

### Example 3 — Optimization in Machine Learning Contexts

Three examples are highlighted:

- Neural network training: minimize a loss function $L(\theta)$ over weights $\theta$.
- SVM: maximize the margin of the separating hyperplane.
- Regression: choose model capacity to balance underfitting and overfitting.

### Example 4 — Device Sizing in Electronic Design

Device sizing chooses transistor widths and lengths under process and timing constraints while minimizing total power consumption.

---

## Concept 3 — The Breadth of Optimization Applications

Optimization is used throughout engineering, operations, logistics, finance, and embedded systems. In many real-time systems, optimization is solved repeatedly as part of a control or design process.

---

## Concept 4 — The Five Phases of an MP/OR Study

1. Problem Definition
2. Model Construction
3. Model Solution
4. Model Validity
5. Implementation

The OR model is informally:

> Maximize or minimize an objective function subject to constraints.

The workflow parallels machine learning:
- Problem definition ↔ data and target selection
- Model construction ↔ choose model class and loss
- Model solution ↔ train the model
- Model validity ↔ validation/test evaluation
- Implementation ↔ deployment

---

## Concept 5 — Formalizing the General Optimization Problem

### Definition

The general constrained optimization problem is:

$$
\min_{\mathbf{x}} f(\mathbf{x})
$$

$$
\text{s.t.} \quad h_i(\mathbf{x}) = 0, \quad i \in I = \{1,\dots,p\}
$$

$$
g_j(\mathbf{x}) \le 0, \quad j \in J = \{1,\dots,m\}
$$

$$
\mathbf{x} \in C
$$

where $\mathbf{x} \in C \subset \mathbb{R}^n$, and $f, h_1,\dots,h_p, g_1,\dots,g_m$ are functions defined on $C$.

### Mathematical Meaning of the Symbols

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | decision variable vector |
| $n$ | number of decision variables |
| $f(\mathbf{x})$ | objective function |
| $h_i(\mathbf{x})$ | equality constraint |
| $g_j(\mathbf{x})$ | inequality constraint |
| $C$ | ambient set |
| $I, J$ | index sets |

### The Feasible Set

$$
F = \{\mathbf{x} \in C \mid h_i(\mathbf{x})=0\ \forall i,\ g_j(\mathbf{x}) \le 0\ \forall j\}
$$

This is the set of all feasible candidates satisfying every constraint.

### Contours of the Objective Function

Level curves satisfy:

$$
\{\mathbf{x}: f(\mathbf{x}) = C\}.
$$

The optimum typically lies at the innermost contour within the feasible region.

### Restating the Problem with Unified Constraint Notation

$$
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad C(\mathbf{x}) \ge 0
$$

Equality and inequality constraints can be rewritten in this canonical form.

### Optimal Value and Optimal Point

$$
p^* = \inf\{ f(\mathbf{x}) \mid h_i(\mathbf{x})=0\ \forall i,\ g_j(\mathbf{x}) \le 0\ \forall j\}
$$

$$
\mathbf{x}^* \in X_{opt}, \quad X_{opt} = \{\mathbf{x} \mid h_i(\mathbf{x})=0\ \forall i,\ g_j(\mathbf{x})\le 0\ \forall j,\ f(\mathbf{x})=p^*\}
$$

A maximization problem may be converted to minimization via:

$$
\max_{\mathbf{x}} f(\mathbf{x}) = -\min_{\mathbf{x}} [-f(\mathbf{x})].
$$

---

## Concept 6 — Active and Inactive Constraints

An inequality constraint is active at a feasible point if

$$
g_j(\mathbf{x}) = 0.
$$

It is inactive if

$$
g_j(\mathbf{x}) < 0.
$$

Equality constraints are always active, since they must hold with equality at every feasible point.

---

## Concept 7 — Single-Variable Optimization: Local and Global Minima

A function $f(x)$ has a local minimum at $x^*$ if

$$
f(x^*) \le f(x^*+h)
$$

for all sufficiently small $h$.

It has a global minimum at $x^*$ if

$$
f(x^*) \le f(x)
$$

for all $x$ in the domain.

---

## Concept 8 — Necessary Condition for a Single-Variable Local Extremum

### Statement

If $f(x)$ has a local minimum at an interior point $x^*$ and $f'(x^*)$ exists, then

$$
f'(x^*) = 0.
$$

### Proof

By definition,

$$
f'(x^*) = \lim_{h\to 0} \frac{f(x^*+h)-f(x^*)}{h}.
$$

Since $x^*$ is a local minimum,

$$
f(x^*) \le f(x^*+h)
$$

for all sufficiently small $h$. Hence,

$$
f(x^*+h)-f(x^*) \ge 0.
$$

If $h>0$, then dividing by a positive number keeps the sign:

$$
\frac{f(x^*+h)-f(x^*)}{h} \ge 0 \quad (h>0)
$$

Therefore,

$$
\lim_{h\to 0^+} \frac{f(x^*+h)-f(x^*)}{h} \ge 0,
$$

which implies

$$
f'(x^*) \ge 0.
$$

If $h<0$, then dividing by a negative number reverses the sign:

$$
\frac{f(x^*+h)-f(x^*)}{h} \le 0 \quad (h<0)
$$

Therefore,

$$
\lim_{h\to 0^-} \frac{f(x^*+h)-f(x^*)}{h} \le 0,
$$

which implies

$$
f'(x^*) \le 0.
$$

Combining both inequalities gives:

$$
f'(x^*) = 0.
$$

### Limitations

1. The theorem fails if $f'(x^*)$ does not exist.
2. The theorem does not apply at endpoints.
3. The converse is false: $f'(x^*)=0$ does not guarantee a local extremum.

Example:

$$
f(x)=x^3,
$$

then

$$
f'(0)=0,
$$

but $x=0$ is not a local extremum.

---

## Concept 9 — Worked Example: Classifying Stationary Points of a Quintic

Consider:

$$
f(x) = 12x^5 - 45x^4 + 40x^3 + 5.
$$

Differentiate:

$$
f'(x) = 60x^4 - 180x^3 + 120x^2.
$$

Factor:

$$
f'(x) = 60x^2(x^2 - 3x + 2) = 60x^2(x-1)(x-2).
$$

So the stationary points are:

$$
x = 0,\ 1,\ 2.
$$

Second derivative:

$$
f''(x) = 240x^3 - 540x^2 + 240x.
$$

Evaluate at each stationary point:

$$
f''(1) = -60 < 0 \quad \Rightarrow \quad \text{local maximum at } x=1,
$$

$$
f''(2) = 240 > 0 \quad \Rightarrow \quad \text{local minimum at } x=2.
$$

At $x=0$, the second derivative test is inconclusive. Compute the third derivative:

$$
f'''(x) = 720x^2 - 1080x + 240,
$$

so

$$
f'''(0) = 240 \ne 0.
$$

Since the first nonzero higher derivative is odd order, $x=0$ is neither a max nor a min; it is an inflection point.

Evaluate the function values:

$$
f(1) = 12,
$$

$$
f(2) = -11.
$$

Thus:
- local maximum value: $12$
- local minimum value: $-11$
- inflection point: $x=0$

---

## Concept 10 — Sufficient Condition for a Single-Variable Local Extremum

### Statement

Let

$$
f'(x^*) = f''(x^*) = \cdots = f^{(n-1)}(x^*) = 0,
$$

but

$$
f^{(n)}(x^*) \ne 0.
$$

Then:
- if $n$ is even and $f^{(n)}(x^*) > 0$, $x^*$ is a local minimum;
- if $n$ is even and $f^{(n)}(x^*) < 0$, $x^*$ is a local maximum;
- if $n$ is odd, $x^*$ is neither a max nor a min.

### Taylor Expansion Idea

$$
f(x^*+h)-f(x^*) = \frac{h^n}{n!} f^{(n)}(x^* + \theta h)
$$

for some $\theta \in (0,1)$.

If $n$ is even, $h^n$ does not change sign. If $n$ is odd, it does.

---

## Concept 11 — Multivariable Optimization: Necessary Condition

### Statement

If $f(\mathbf{x})$ has a local minimum or maximum at $\mathbf{x}^*$ and the first partial derivatives exist, then

$$
\nabla f(\mathbf{x}^*) = 0.
$$

### Gradient

The gradient is

$$
\nabla f(\mathbf{x}) = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right).
$$

### Proof

Use the first-order Taylor expansion:

$$
f(\mathbf{x}^*+\mathbf{h}) - f(\mathbf{x}^*) = \mathbf{h}^T \nabla f(\mathbf{x}^*) + R_1(\mathbf{x}^*,\mathbf{h}).
$$

If $\nabla f(\mathbf{x}^*) \ne 0$, choose $\mathbf{h} = -\nabla f(\mathbf{x}^*)$ so that

$$
\mathbf{h}^T \nabla f(\mathbf{x}^*) < 0.
$$

Then, for sufficiently small $\mathbf{h}$, this negative linear term dominates the remainder, and we get

$$
f(\mathbf{x}^*+\mathbf{h}) < f(\mathbf{x}^*),
$$

which contradicts local minimality.

Thus,

$$
\nabla f(\mathbf{x}^*) = 0.
$$

---

## Concept 12 — Multivariable Optimization: Sufficient Condition (Hessian Test)

The Hessian is

$$
H(\mathbf{x}) = \nabla^2 f(\mathbf{x})
$$

with entries

$$
H_{ij}(\mathbf{x}) = \frac{\partial^2 f}{\partial x_i \partial x_j}.
$$

If $\mathbf{x}^*$ is stationary and the Hessian is positive definite, then $\mathbf{x}^*$ is a local minimum.

A symmetric matrix $A$ is positive definite if any of the following equivalent conditions holds:

- all eigenvalues are positive;
- all leading principal minors are positive;
- $\mathbf{h}^T A \mathbf{h} > 0$ for all $\mathbf{h}\ne 0$.

The Hessian test says:

$$
H(\mathbf{x}^*) \succ 0 \quad \Rightarrow \quad \text{local minimum}.
$$

For a saddle point, the Hessian is indefinite.

---

## Concept 13 — Semi-Definite Case and Saddle Points

If the Hessian is positive semi-definite,

$$
H(\mathbf{x}^*) \succeq 0,
$$

the second-order test is inconclusive.

Example:

$$
f(x,y) = x^2 - y^2.
$$

Its gradient is

$$
\nabla f(x,y) = (2x, -2y),
$$

and the Hessian is

$$
H = \begin{pmatrix} 2 & 0 \\ 0 & -2 \end{pmatrix}.
$$

This matrix has mixed eigenvalues, so the point is a saddle point.

---

## Concept 14 — Equality-Constrained Optimization

The equality-constrained problem is:

$$
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad h_j(\mathbf{x}) = 0.
$$

Named methods include:
- direct substitution,
- constrained variation,
- Lagrange multipliers.

---

## Concept 15 — Inequality-Constrained Optimization

The inequality-constrained problem is:

$$
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_j(\mathbf{x}) \le 0.
$$

The main named method is the Kuhn–Tucker (KKT) conditions.

---

## Concept 16 — Classification of Optimization Problems

### 16.1 Linear Programming (LP)

$$
\min_{\mathbf{x}} \mathbf{C}^T\mathbf{x} \quad \text{s.t.} \quad \mathbf{A}\mathbf{x} = \mathbf{b}.
$$

### 16.2 Unconstrained Optimization

$$
\min_{\mathbf{x}} f(\mathbf{x}).
$$

### 16.3 Convex Optimization

A convex function satisfies:

$$
f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \le \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y}).
$$

### 16.4 Quadratic Optimization

$$
f(\mathbf{x}) = a + \mathbf{b}^T\mathbf{x} + \mathbf{x}^T H \mathbf{x}.
$$

### 16.5 Geometric Programming

Posynomial form:

$$
f(\mathbf{x}) = \sum_{k=1}^{K} c_k x_1^{r_{1k}}x_2^{r_{2k}}\cdots x_n^{r_{nk}}.
$$

### 16.6 Integer Programming

$$
x_i \in \mathbb{Z}.
$$

### 16.7 Stochastic Programming

The objective is often an expected value:

$$
\mathbb{E}[f(\mathbf{x},\xi)].
$$

### 16.8 Multi-objective Optimization

A common scalarization is:

$$
f(\mathbf{x}) = \alpha_1 f_1(\mathbf{x}) + \alpha_2 f_2(\mathbf{x}).
$$

### 16.9 Modern/Heuristic Methods

Examples:
- genetic algorithms,
- simulated annealing,
- particle swarm optimization,
- ant colony optimization.

### 16.10 Deep Learning Algorithms

Examples:
- gradient descent,
- SGD,
- momentum,
- AdaGrad,
- RMSProp,
- Adam.

---

# Mathematical Foundations (Consolidated)

### The Gradient

$$
\nabla f(\mathbf{x}) = \left(\frac{\partial f}{\partial x_1},\dots,\frac{\partial f}{\partial x_n}\right).
$$

### The Hessian

$$
\nabla^2 f(\mathbf{x})_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}.
$$

### Taylor Expansions

Single-variable form:

$$
f(x^*+h)-f(x^*) = \frac{h^n}{n!}f^{(n)}(x^*+\theta h).
$$

Multivariable first-order form:

$$
f(\mathbf{x}^*+\mathbf{h})-f(\mathbf{x}^*) = \mathbf{h}^T\nabla f(\mathbf{x}^*) + R_1(\mathbf{x}^*,\mathbf{h}).
$$

Multivariable second-order form at a stationary point:

$$
f(\mathbf{x}^*+\mathbf{h})-f(\mathbf{x}^*) = \frac{1}{2}\mathbf{h}^T\nabla^2 f(\mathbf{x}^*+\theta\mathbf{h})\mathbf{h}.
$$

---

# Problem-Solving Procedures

### Procedure A — Classify Stationary Points of a Single-Variable Function

1. Compute $f'(x)$.
2. Solve $f'(x)=0$.
3. Compute $f''(x^*)$.
4. If $f''(x^*)=0$, continue to higher derivatives.
5. Use the first nonzero derivative to classify the point.

### Procedure B — Classify Stationary Points of a Multivariable Function

1. Compute $\nabla f(\mathbf{x})$.
2. Solve $\nabla f(\mathbf{x})=0$.
3. Compute $H(\mathbf{x}^*)$.
4. Test definiteness:
   - positive definite → local minimum,
   - negative definite → local maximum,
   - indefinite → saddle point.

### Procedure C — Solve an Equality-Constrained Problem by Direct Substitution

1. Identify objective and constraint.
2. Solve the constraint for one variable.
3. Substitute into the objective.
4. Optimize the reduced problem.
5. Substitute back to recover the original variables.

---

# Comparisons

| Concept A | Concept B | Key Distinction |
|---|---|---|
| Local optimum | Global optimum | Local = best nearby; global = best over whole domain |
| Necessary condition | Sufficient condition | Necessary must hold; sufficient guarantees result |
| Active constraint | Inactive constraint | Active means equality; inactive means slack |
| Equality constraint | Inequality constraint | Equality exactly satisfies the constraint |
| Positive definite | Positive semi-definite | PD is strict; PSD allows zero directions |
| Convex function | Convex set | Function graph vs. set geometry |
| LP | NLP | LP has affine functions only |
| Classical methods | Heuristic methods | Derivative-based vs. derivative-free |
| Saddle point | Local extremum | Increases in some directions and decreases in others |

---

# AI / Machine Learning Connections

| Optimization concept | Application |
|---|---|
| Gradient = 0 | Neural network training |
| Quadratic optimization | SVM margins |
| Bias–variance trade-off | Model capacity selection |
| Convexity | Reliable training objectives |
| SGD | Stochastic optimization |

---

# Common Mistakes / Exam Traps

1. Assuming $f'(x^*)=0$ guarantees an extremum.
2. Applying the theorem where the derivative does not exist.
3. Applying the theorem at an endpoint.
4. Forgetting to check earlier derivatives in the higher-order test.
5. Confusing PSD with PD.
6. Treating local and global extrema as identical.
7. Assuming touching a boundary means infeasible.
8. Forgetting that equality constraints are always active.

---

# Exam Preparation

## What You Must Understand

- definition of optimization and OR
- five phases of an MP/OR study
- decision variables, objective, constraints, feasible set
- active vs. inactive constraints
- local vs. global extrema
- necessary vs. sufficient conditions
- saddle points
- LP, NLP, convex, quadratic, GP, integer, stochastic, and multi-objective optimization

## What You Must Be Able to Derive

- proof that $f'(x^*)=0$ is necessary for a local minimum
- proof of the higher-order sufficient condition
- proof that $\nabla f(\mathbf{x}^*)=0$ is necessary in the multivariable case
- proof that a positive definite Hessian implies a local minimum

## What You Must Be Able to Calculate

- find and classify stationary points of one-variable functions
- find and classify stationary points of multivariable functions
- test definiteness of symmetric matrices
- solve simple equality-constrained problems by substitution

---

# Formula Sheet

| # | Formula | Interpretation |
|---|---|---|
| 1 | $\min_{\mathbf{x}} f(\mathbf{x})$ s.t. constraints | general optimization problem |
| 2 | $F=\{\mathbf{x}\in C \mid \text{constraints}\}$ | feasible set |
| 3 | $p^*=\inf\{f(\mathbf{x})\mid \mathbf{x}\in F\}$ | optimal value |
| 4 | $f'(x^*)=0$ | necessary condition (1D) |
| 5 | $\nabla f(\mathbf{x}^*)=0$ | necessary condition (multiD) |
| 6 | $H(\mathbf{x}^*)\succ0$ | sufficient condition for local minimum |
| 7 | $\mathbf{h}^TA\mathbf{h}>0$ for all $\mathbf{h}\ne0$ | positive definite matrix |
| 8 | $\min_{\mathbf{x}} \mathbf{C}^T\mathbf{x}$ s.t. $\mathbf{A}\mathbf{x}=\mathbf{b}$ | linear program |
| 9 | $f(\mathbf{x})=a+\mathbf{b}^T\mathbf{x}+\mathbf{x}^T H \mathbf{x}$ | quadratic objective |
| 10 | $x_i\in\mathbb{Z}$ | integer restriction |

---

# Glossary

- Active constraint: an inequality constraint that holds with equality.
- Affine function: linear function plus a constant.
- Constraint: a condition a feasible solution must satisfy.
- Convex function: a function whose graph lies below its chords.
- Decision variable: a quantity chosen by the optimizer.
- Feasible set: all points satisfying all constraints.
- Gradient: vector of first partial derivatives.
- Hessian: matrix of second partial derivatives.
- Local minimum: best value in a neighborhood.
- Global minimum: best value over the whole domain.
- Saddle point: point with indefinite Hessian and mixed directional behavior.
- Stationary point: point where the gradient or derivative is zero.
- Sufficient condition: condition that guarantees an extremum type.

---

# Self-Assessment Questions

1. Write the general constrained optimization problem in full notation.
2. Explain the difference between feasible set and optimal set.
3. Prove that $f'(x^*)=0$ is necessary for a local minimum.
4. Give an example where $f'(x^*)=0$ but there is no local extremum.
5. State the multivariable necessary condition and give a proof sketch.
6. Classify the matrix $\begin{pmatrix}-2&0\\0&-5\end{pmatrix}$.
7. Find the stationary points of $f(x)=x^4$.
8. Why are discrete problems sometimes solved by enumeration instead of calculus?

---

# Final Review

This chapter established the conceptual and mathematical framework for optimization:

- optimization is the science of allocating limited resources under constraints;
- the general optimization problem is formalized using variables, objective, and constraints;
- local extrema are identified using first-order conditions and classified using higher-order or Hessian tests;
- optimization problems are classified by their mathematical structure;
- the concepts introduced here are the foundation for later work in constrained optimization, convexity, and machine learning.

In short, the chapter teaches the language of optimization and the calculus tools used to understand it.
