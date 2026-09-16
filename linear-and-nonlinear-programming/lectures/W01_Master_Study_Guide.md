# Math 303 — Linear and Nonlinear Programming for Computational Sciences
## Master Study Guide — Week 1: Fundamentals of Optimization & Classical Optimization Techniques

---

## Overview

This chapter builds the mathematical and conceptual foundation for the entire course. It answers three questions:

1. **What is optimization, and why does it matter for engineering, science, and Artificial Intelligence?**
2. **How do we formally state an optimization problem?**
3. **How do we recognize an optimal point mathematically — for a function of one variable, and for a function of several variables?**

The lecture does two things. First, it introduces optimization as a discipline (*Fundamentals*): what it is, how it is classified, how a study is carried out in practice, and how the general optimization problem is written mathematically. Second, it introduces the *Classical Optimization Techniques* — the calculus-based machinery (derivatives, gradients, Hessians) used to test whether a candidate point is actually a minimum, a maximum, or neither.

This guide teaches every statement, theorem, proof, example, and classification in the source lecture in full, and — where genuinely necessary for understanding — adds explanatory material, worked derivations, and connections to Artificial Intelligence / Machine Learning. Material that is **not** in the original slides is explicitly labeled **Additional Explanation** or **Example (Added)** so you always know what came from the lecture and what was added to help you understand it.

---

## Learning Objectives

By the end of this chapter you should be able to:

1. Define optimization (mathematical programming) and explain its role in Operations Research.
2. Reproduce Table 1.1 (methods of Operations Research) and place any given technique into the correct category.
3. Carry out the five phases of an OR/MP study on a new problem.
4. Write the general constrained optimization problem in standard form, identify decision variables, objective function, equality/inequality constraints, and the feasible set.
5. Distinguish active vs. inactive inequality constraints, and explain why equality constraints are always active.
6. State and prove the **necessary condition** for a local extremum of a single-variable function, and explain its three limitations (non-differentiability, endpoints, converse failure).
7. State and prove the **sufficient condition** for a local extremum of a single-variable function using higher-order derivatives, and apply it to classify stationary points.
8. State and prove the **necessary condition** ($\nabla f(\mathbf{x}^*) = 0$) and the **sufficient condition** (Hessian positive definite) for a local minimum of a multivariable function.
9. Determine whether a symmetric matrix is positive definite, positive semi-definite, or indefinite, and connect this to local minima, local maxima, and saddle points.
10. Classify an optimization problem into its correct family: Linear Programming, Unconstrained Optimization, Convex Optimization, Quadratic Optimization, Geometric Programming, Integer Programming, Stochastic Programming, Multi-objective Optimization, or Modern (heuristic) Optimization.
11. Connect every one of the above ideas to a concrete AI/ML example: neural network training, SVM margin maximization, and the bias–variance/generalization trade-off in regression.

---

## Prerequisites

You need the following background to follow this chapter. Where the lecture assumes it silently, this guide supplies it.

- **Single-variable calculus:** derivatives, higher-order derivatives, Taylor's theorem with remainder.
- **Multivariable calculus:** partial derivatives, the gradient vector, the Hessian matrix, multivariable Taylor expansion.
- **Linear algebra:** vectors, matrices, matrix–vector and matrix–matrix multiplication, eigenvalues, quadratic forms ($\mathbf{h}^T A \mathbf{h}$), determinants (specifically leading principal minors).
- **Basic real analysis vocabulary:** limit, neighborhood, interval, domain of a function.
- **Set notation:** $\in$, $\subset$, $\{\cdot \mid \cdot\}$.

---

# Main Concepts

## Concept 1 — What Is Optimization?

### Definition

> **From the Lecture:** Optimization (Programming) is the science that is concerned with the efficient use (allocation) of limited resources to meet certain objectives and satisfy predefined constraints (limitations). Optimization is the act of obtaining the best result under given circumstances. It is also called **Mathematical Programming (MP)** and is generally studied as a part of **Operations Research (OR)**.

### Intuition

Every optimization problem has the same skeleton: **you have choices to make (variables), you have limited resources or rules you must respect (constraints), and you have a single yardstick for "better" (the objective).** Optimization is the systematic search for the choice that is best according to that yardstick, without breaking the rules.

### Why It Exists

**From the Lecture:** In design, construction, and maintenance of any engineering system, engineers must take many technological and managerial decisions at several stages. The ultimate goal of all such decisions is either to **minimize the effort (cost)** required or to **maximize the desired benefit (profit)**.

**Additional Explanation:** Without a formal optimization framework, engineers would be forced to compare candidate designs one at a time, by trial and error, with no guarantee that a better design isn't still undiscovered. Mathematical programming replaces this guesswork with a structured search that can, under the right conditions (see *Convexity*, later), guarantee that the solution found really is the best possible one — not merely "good."

### Table 1.1 — Methods of Operations Research

**From the Lecture:**

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

**Modern or nontraditional optimization techniques (from the Lecture):**
- Genetic algorithms
- Simulated annealing
- Ant colony optimization
- Particle swarm optimization
- Neural networks
- Fuzzy optimization

**Additional Explanation:** This table shows that "Operations Research" is broader than "Optimization" — it also includes stochastic process techniques (for randomness over time) and statistical methods (for data analysis). This course, Math 303, focuses on the **left column and the modern techniques row**: mathematical programming and (later, in other lectures) heuristic search. Note that this course's title — *Linear and Nonlinear Programming* — corresponds to two specific entries in this table's left column.

### AI / Machine Learning Connection

Nearly every training algorithm in machine learning is literally an instance of "Calculus methods" or "Nonlinear programming" from this table: training a neural network is a large-scale, generally non-convex, unconstrained (or lightly constrained) nonlinear optimization problem. "Neural networks" itself appears in the table as a *modern optimization technique* — historically, neural networks were originally studied partly as heuristic optimizers/pattern-associators before becoming the dominant technique for supervised learning.

---

## Concept 2 — Motivating Examples of Optimization (From the Lecture)

The lecture uses four worked examples to show that "optimization" is not one narrow technique but a way of thinking that appears in everyday decisions, geometry, engineering, and machine learning.

### Example 1 — Minimizing the Cost of Business-Trip Airfare

**Problem statement (from the Lecture):**
You have a business trip between CAI (Cairo) and Dubai. You fly out every Monday and return every Wednesday, for 5 weeks. A regular round-trip ticket costs \$400, but a 20% discount is granted if the ticket spans a weekend. A one-way ticket costs 75% of the regular price. How should you buy the tickets for the 5-week period?

**Solution — Problem Definition (From the Lecture):**

- **Restrictions:** You must go on Monday and return on Wednesday of the *same* week (a hard business requirement — this is what makes the problem non-trivial).
- **Objective criterion:** total ticket cost.

**Decision alternatives identified in the lecture:**

1. **Alternative 1:** Buy 5 separate regular round trips (CAI→Dubai→CAI, Monday to Wednesday, each week).
2. **Alternative 2:** Buy 1 one-way CAI→Dubai ticket (Monday, week 1), then 4 *regular round trips* but shifted to span the weekend (Dubai→CAI→Dubai, Wednesday to Monday), then 1 one-way Dubai→CAI ticket (Wednesday, week 5).
3. **Alternative 3:** Buy 1 regular round trip spanning the *entire* 5 weeks (CAI Monday of week 1 → Dubai → CAI Wednesday of week 5) **plus** 4 regular round trips that span the weekend (Dubai→CAI→Dubai, Wednesday to Monday).

**Additional Explanation (why these three and not others):** Notice the trick embedded in the problem: a round trip is cheaper *only if it spans a weekend* (Fri/Sat/Sun somewhere inside it). Since the business requirement fixes Monday-out/Wednesday-back within a week, a *plain* week's round trip (Mon→Wed) never spans a weekend and therefore never earns the discount. The only way to capture the 20% discount is to restructure which city you are "based" in during the weekend — i.e., treat some of your round trips as Dubai→CAI→Dubai (Wed to Mon) instead of CAI→Dubai→CAI (Mon to Wed). This restructuring is exactly what Alternatives 2 and 3 do; only the "bookkeeping" of the two endpoint one-way legs differs.

**Cost calculations (from the Lecture):**

$$
\text{Alternative 1 cost} = 5 \times 400 = \$2000
$$

$$
\text{Alternative 2 cost} = 0.75\times 400 + 4\times(0.8\times 400) + 0.75\times 400 = \$1880
$$

$$
\text{Alternative 3 cost} = 5 \times (0.8\times 400) = \$1600
$$

**Additional Explanation (verifying the arithmetic):**
- Alternative 1: $5 \times 400 = 2000$. ✓
- Alternative 2: $0.75\times400 = 300$ (one-way leg); $0.8\times400=320$ (discounted round trip). So $300 + 4(320) + 300 = 300+1280+300 = 1880$. ✓
- Alternative 3: $0.8 \times 400 = 320$ per discounted round trip, times 5 trips $= 1600$. ✓

**Conclusion:** Alternative 3 is cheapest (\$1600) — it earns the weekend discount on *every* one of the five round trips, with no full-price one-way legs at all. This illustrates the essence of optimization: among a *finite* set of feasible alternatives, pick the one with the best objective value.

**AI / Machine Learning Connection:** This example is a **discrete/combinatorial optimization problem** — the "decision variable" is a choice among a small, enumerable set of travel plans, not a continuous quantity. This mirrors *hyperparameter search* in ML (e.g., "should I use Adam, SGD, or RMSProp?") where you enumerate a finite set of configurations and pick the one that minimizes validation loss or cost.

---

### Example 2 — Maximum-Area Rectangle from a Fixed-Length Wire

**Problem statement (from the Lecture):** What are the length and width of the rectangle of maximum area formed from a piece of wire of length $L$ inches?

**Solution (from the Lecture):**

- **Variables:** rectangle width $w$ and length $h$.
- **Objective:** maximize Area $= w \times h$.
- **Restrictions:** $L = 2(w+h)$, $w>0$, $h>0$.
- **Result (via calculus):** $w = h = L/4$.

**Additional Explanation — full derivation (this is what "solve using calculus" means, worked explicitly):**

This is a **constrained optimization problem with an equality constraint**. The lecture's own later section ("Solution with direct substitution") tells you exactly how to solve it: use the constraint to eliminate one variable, then apply single-variable calculus.

From the constraint $L = 2(w+h)$, solve for $h$:
$$
h = \frac{L}{2} - w
$$

Substitute into the objective:
$$
A(w) = w\left(\frac{L}{2} - w\right) = \frac{L}{2}w - w^2
$$

This is now an **unconstrained single-variable** maximization problem in $w$ alone (with the implicit domain $0 < w < L/2$ so that $h>0$ too). Apply the necessary condition (Concept 5 below): set the first derivative to zero.

$$
A'(w) = \frac{L}{2} - 2w = 0 \implies w = \frac{L}{4}
$$

Then:
$$
h = \frac{L}{2} - \frac{L}{4} = \frac{L}{4}
$$

So $w = h = L/4$, confirming the lecture's stated result — **the maximum-area rectangle for a fixed perimeter is a square.**

Checking the sufficient condition: $A''(w) = -2 < 0$, so this stationary point is indeed a **maximum** (matches the theorem in Concept 6 with $n=2$ even and $f^{(2)}<0$).

**Geometric Interpretation:** As $w$ ranges over $(0, L/2)$, $A(w)$ is a downward-opening parabola; its unique maximum is at the vertex $w=L/4$. This is the simplest possible illustration of "local max = global max," because a strictly concave (downward-curving) function on an interval has only one stationary point, and it is automatically the global maximum on that interval.

---

### Example 3 — Optimization in Machine Learning Contexts (From the Lecture)

The lecture lists three ML-flavored appearances of optimization, without derivation (they are previews of topics developed later in the course):

**(a) Adjusting weights during NN (neural network) training.**

**Additional Explanation:** Training a neural network means choosing the weight vector $\theta$ that minimizes a loss function $L(\theta)$ measuring the mismatch between predictions and true labels over a training set. This is exactly the *unconstrained multivariable optimization problem* of Concept 7/8 below, except that $f = L$ is a function of possibly millions of variables (the weights), and is typically **non-convex**, so only *local* minima can be guaranteed by classical calculus-based conditions.

**(b) Optimal hyperplane in SVM (Support Vector Machine).**

**Additional Explanation:** In the lecture's figure, points of one class (squares) and another class (circles) are separated by a line ("optimal hyperplane") with maximum margin $\rho_0$ to the nearest points of either class (the "support vectors," marked $\mathbf{x}^{(s)}$). Finding this hyperplane is a **constrained optimization problem**: maximize the margin subject to every training point being correctly classified. As you will see later in the course, this turns out to be a **Quadratic Optimization** problem in the sense of Concept 15 below (quadratic objective, affine/linear constraints) — one of the cleanest real-world examples of a convex quadratic program.

**(c) Regression Models — the Underfitting/Appropriate-Capacity/Overfitting Trade-off.**

**From the Lecture (figure description):** Three regimes are shown for a regression model fit to noisy data points:
- **Underfitting** — the fitted curve (here, a straight line) is too simple to capture the true pattern in the data.
- **Appropriate capacity** — the fitted curve matches the true trend well.
- **Overfitting** — the fitted curve is so flexible that it passes through every data point, including noise.

The accompanying **Error vs. Capacity** plot shows:
- **Training error** (dashed): decreases monotonically as model capacity increases (a more flexible model can always fit the training data better, or at least no worse).
- **Generalization error** (solid): decreases, reaches a minimum, then increases again.
- **Underfitting zone:** left of the **Optimal Capacity** point.
- **Overfitting zone:** right of the **Optimal Capacity** point.
- **Generalization gap:** the vertical distance between the generalization-error curve and the training-error curve at a given capacity.

**Additional Explanation — why this belongs in an optimization course:** Fitting a regression model is itself an optimization problem: minimize training error (a loss function) over the model's parameters. But the plot shows that **minimizing training error is not the true goal** — the true goal is minimizing *generalization* error (error on unseen data), and these two objectives diverge once the model is complex enough to overfit. This is why later in ML you optimize a **regularized objective** (training loss *plus* a penalty term controlling complexity) instead of the raw training loss — regularization is a way of steering the unconstrained optimization problem toward the "Optimal Capacity" point rather than letting training error alone run away toward zero.

---

### Example 4 — Device Sizing in Electronic Design (From the Lecture)

**From the Lecture:**
- Device sizing is the task of choosing the width and length of each device (transistor) in an electronic circuit.
- **Variables:** the widths and lengths.
- **Constraints:** engineering requirements such as (i) limits on device sizes imposed by the manufacturing process, (ii) timing requirements ensuring the circuit operates reliably at a specified speed, and (iii) a limit on the total circuit area.
- **A common objective:** the total power consumed by the circuit.

**Additional Explanation:** This example is included by the instructor specifically because it is a classic illustration of **Geometric Programming** (Concept 16 below): circuit power, area, and delay are frequently well-approximated by posynomial functions of the device dimensions, so device sizing is one of the textbook applications of GP in engineering.

---

## Concept 3 — The Breadth of Optimization Applications (From the Lecture)

**From the Lecture:**
> An amazing variety of practical problems involving decision making can be cast in the form of a mathematical optimization problem. It is widely used in engineering — electronic design automation, automatic control systems, network design and operation, finance, supply chain management, scheduling, civil/chemical/mechanical/aerospace engineering. With the proliferation of computers embedded in products, we have seen rapid growth in **embedded optimization**, where optimization automatically makes real-time choices with little or no human intervention.

**Additional Explanation:** "Embedded optimization" is worth pausing on: it is the idea that an optimization *solver* itself becomes part of a running system (e.g., a car's cruise control repeatedly solving a small optimization problem every fraction of a second to decide throttle position, or a recommender system solving an optimization problem for every user request). This is conceptually the bridge between classical, "solve it once on paper/offline" optimization and modern, real-time AI systems.

---

## Concept 4 — The Five Phases of an MP/OR Study

**From the Lecture:**

1. **Problem Definition:** define variables, objective function (OF), constraints.
2. **Model Construction:** put the problem in mathematical form.
3. **Model Solution:** use an optimization algorithm — apply MP techniques and Sensitivity Analysis.
4. **Model Validity:** does the solution make sense? Is it acceptable? Compare with historical data.
5. **Implementation:** issue operational instructions.

**The OR model (informal statement, from the Lecture):**
> Max (Min) objective function, subject to Constraints.

### Why It Exists

**Additional Explanation:** This five-phase structure is the "engineering method" applied to optimization — it exists because a mathematically optimal solution to a *wrong model* is worthless. Phase 4 ("Model Validity") is frequently skipped by beginners, who treat "the math worked out" as equivalent to "the answer is correct." In reality, if a model's assumptions don't match reality (e.g., you assumed linear costs but costs are actually stepped, as in the airfare example), the "optimal" answer may not be implementable or may be far from actually optimal in the real world.

### AI / Machine Learning Connection

This maps directly onto the machine learning workflow:
- **Problem Definition** ↔ deciding what to predict and what data/features are available.
- **Model Construction** ↔ choosing a model class (linear regression, neural network, etc.) and a loss function.
- **Model Solution** ↔ training (running gradient descent or another optimizer).
- **Model Validity** ↔ validation/test-set evaluation, checking for overfitting (see Example 3(c) above).
- **Implementation** ↔ deployment of the trained model.

---

## Concept 5 — Formalizing the General Optimization Problem

### Definition

**From the Lecture — the general optimization problem:**

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

### Mathematical Meaning — Explaining Every Symbol

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | The **decision variable** vector — the quantity the optimizer is free to choose, $\mathbf{x} = (x_1,\dots,x_n)$. |
| $n$ | The number of decision variables (the dimensionality of the search space). |
| $f(\mathbf{x})$ | The **objective function** — the single scalar quantity we want to minimize (cost, error, negative profit, etc.). |
| $h_i(\mathbf{x})$ | The $i$-th **equality constraint function**; the requirement $h_i(\mathbf{x})=0$ must hold exactly. |
| $p$ | The number of equality constraints. |
| $g_j(\mathbf{x})$ | The $j$-th **inequality constraint function**; the requirement $g_j(\mathbf{x}) \le 0$ must hold. |
| $m$ | The number of inequality constraints. |
| $C$ | An underlying "ambient" set (e.g., $\mathbb{R}^n$, or a box like $x_i \ge 0$) on which all the functions are defined. |
| $I, J$ | Index sets simply enumerating which equality/inequality constraint is which ($I=\{1,\dots,p\}$, $J=\{1,\dots,m\}$). |

**Additional Explanation — decision variables vs. parameters:** It is important to distinguish the decision variables $\mathbf{x}$ (what the optimizer *chooses*) from **parameters** — fixed numbers that define the problem instance but are not being optimized over (e.g., in Example 2 above, $L$ is a parameter; $w,h$ are decision variables). The lecture's general formula does not show parameters explicitly, but every constraint and objective function in a real problem is built from a combination of decision variables and fixed parameters/data.

### The Feasible Set

**From the Lecture:**
$$
F = \{ \mathbf{x} \in C \mid h_i(\mathbf{x})=0\ \forall i,\ g_j(\mathbf{x}) \le 0\ \forall j \}
$$

**Intuition:** $F$ is simply the collection of *all* candidate points that satisfy every rule of the problem — every equality constraint exactly, and every inequality constraint at-or-below its bound. A point $\mathbf{x}$ is called **feasible** if $\mathbf{x} \in F$, and **infeasible** otherwise. Optimization only ever searches over $F$; points outside $F$ are not valid answers no matter how good their objective value looks.

### Geometric Interpretation — The Design Space Diagram

**From the Lecture** (a hypothetical two-dimensional design space): the diagram shows

- an **infeasible region** (violates at least one constraint),
- a **feasible region** (satisfies all constraints) containing a labeled **free point** (an interior feasible point, touching no constraint boundary),
- curved **behavior constraints** $g_1=0, g_2=0, g_4=0$ (nonlinear constraint boundaries),
- straight **side constraints** $g_3=0, g_5=0$ (often simple bound constraints like $x_i \ge 0$ or $x_i \le \text{const}$),
- a **free unacceptable point** — lies outside the feasible region entirely,
- a **bound acceptable point** — lies exactly on a constraint boundary and *is* feasible (the constraint is active, but not violated),
- a **bound unacceptable point** — lies on one constraint's boundary but is still infeasible because it violates a *different* constraint.

**Additional Explanation:** This last distinction — bound *acceptable* vs. bound *unacceptable* — is the geometric illustration of "active" constraints (Concept 9 below): touching a constraint boundary does not by itself make a point infeasible; a point is infeasible only if it actually *violates* some constraint (equality $\ne 0$, or inequality $>0$).

### Contours of the Objective Function

**From the Lecture:** nested contour (level) curves $f=C_1, f=C_2,\dots,f=C_7$ with $C_1<C_2<\dots<C_7$, drawn inside the feasible region. The **optimum point** sits at the center of the innermost contour ($f=C_1$).

**Additional Explanation — what a contour/level curve is:** A **level curve** (or level set) of $f$ at value $C$ is the set $\{\mathbf{x} : f(\mathbf{x}) = C\}$ — every point that gives the objective function the *same* value. For a minimization problem, contours shrink toward the point of lowest objective value, like the rings of a topographic map shrinking toward a hilltop (except here they shrink toward a *valley bottom*, since we're minimizing). The optimal point in a constrained problem is where the *smallest* feasible contour just touches the feasible region — often, as in this figure, at a point where a contour is tangent to a constraint boundary. This tangency condition is exactly what the method of Lagrange multipliers (named later in the lecture) formalizes algebraically.

### Restating the Problem with a Single Unified Constraint Notation

**From the Lecture:**
$$
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad C(\mathbf{x}) \ge 0
$$

where $C(\mathbf{x})$ represents the whole set of constraints (note: here $C(\mathbf{x})$ denotes a *constraint function*, distinct from the ambient set $C$ used earlier — the lecture reuses the letter $C$ in two different roles).

- **Equality constraints rewritten:** $h_i(\mathbf{x}) = 0 \equiv h_i(\mathbf{x}) \ge 0 \ \text{and}\ -h_i(\mathbf{x}) \ge 0$.
- **Inequality constraints rewritten:** $g_j(\mathbf{x}) \le 0 \equiv -g_j(\mathbf{x}) \ge 0$.

**Additional Explanation — why bother rewriting constraints this way:** This shows that *every* type of constraint (equality or inequality, $\le$ or $\ge$) can be rewritten in a single canonical form, "(something) $\ge 0$." This unification is not just cosmetic — it is exactly what lets general-purpose optimization theory (and general-purpose software solvers) treat every constraint with the same machinery instead of writing separate code/theory for each constraint type. An equality constraint becomes "two inequality constraints back to back," which is intuitive: requiring $h_i(\mathbf{x})=0$ is the same as requiring it to be *simultaneously* $\ge 0$ and $\le 0$.

### The Optimal Value and the Optimal Point

**From the Lecture:**

$$
p^* = \inf\{ f(\mathbf{x}) \mid h_i(\mathbf{x})=0\ \forall i,\ g_j(\mathbf{x}) \le 0\ \forall j\}
$$

$$
\mathbf{x}^* \in X_{opt}, \quad X_{opt} = \{\mathbf{x} \mid h_i(\mathbf{x})=0\ \forall i,\ g_j(\mathbf{x})\le 0\ \forall j,\ f(\mathbf{x})=p^*\}
$$

**Mathematical Meaning:** $p^*$ is the *best achievable value* of the objective function over the feasible set — the infimum ("greatest lower bound"), which is used instead of "minimum" because in some problems the exact minimum value may not actually be attained by any feasible point (e.g., it is only approached in a limit). $X_{opt}$ is the *set* of all feasible points that actually achieve this best value $p^*$ — there could be one such point, many (e.g., a whole edge or region), or, in the infimum case, none at all.

> **From the Lecture — reflection prompt:** *"Think about Maximization problem?!"*
>
> **[UNCLEAR IN SOURCE — DO NOT INFER: the lecture poses this as an open question without providing the worked answer.]**
>
> **Additional Explanation (answering the spirit of the prompt):** Every maximization problem can be converted to the minimization form given above via the identity $\max_{\mathbf{x}} f(\mathbf{x}) = -\min_{\mathbf{x}} [-f(\mathbf{x})]$. So $p^*_{\max} = -\inf\{-f(\mathbf{x})\} = \sup\{f(\mathbf{x})\}$, and all of the machinery above (feasible set, active constraints, etc.) applies unchanged — you simply negate the objective, solve the equivalent minimization problem, and negate the optimal value back at the end. This is why most optimization textbooks (including the lecture's own References list) state theory exclusively in terms of minimization without loss of generality.

---

## Concept 6 — Active and Inactive Constraints, Locally vs. Globally Optimal Points

### Definition

**From the Lecture:**
- If $\mathbf{x}$ is feasible and $g_j(\mathbf{x}) = 0$, the $j$-th inequality constraint is **active** at $\mathbf{x}$.
- If $g_j(\mathbf{x}) < 0$, the $j$-th inequality constraint is **inactive** at $\mathbf{x}$.
- **All equality constraints $h_i(\mathbf{x})=0$ are active at every feasible point.**
- The lecture also flags, as a key distinction to master: **locally versus globally optimal points**, and **feasibility versus optimality conditions.**

### Intuition

"Active" means the constraint is "pressing against" the point — the point sits exactly on the boundary that the constraint defines. "Inactive" means there is slack; the point is strictly inside the region that the constraint allows, and the constraint has (locally) no effect on where the optimum can move. This is why equality constraints are always active: by definition they permit *no* slack at all — a feasible point must sit exactly on the equality surface, always.

### Why It Matters

**Additional Explanation:** Whether a constraint is active turns out to be central to constrained optimization theory (developed further in this course, e.g., in the KKT conditions named later in the lecture). Roughly: an *inactive* inequality constraint has no bearing on the local behavior of the optimal point (you could remove it and the local optimum wouldn't move), whereas an *active* inequality constraint behaves, locally, just like an equality constraint — it genuinely restricts the directions in which you're allowed to move away from $\mathbf{x}^*$.

---

## Concept 7 — Single-Variable Optimization: Local and Global Minima

### Definition

**From the Lecture:**
- $f(x)$ has a **local (relative) minimum** at $x=x^*$ if $f(x^*) \le f(x^*+h)$ for all sufficiently small positive and negative $h$ (i.e., in a neighborhood of $x^*$, as $h \to 0$).
- $f(x)$ has a **global (absolute) minimum** at $x^*$ if $f(x^*) \le f(x)$ for **all** $x \in \text{dom}(f)$.

### Mathematical Meaning

**Additional Explanation:** The distinction is entirely about the *size of the neighborhood being compared*. A local minimum only needs to beat its immediate neighbors (an arbitrarily small interval around it); a global minimum needs to beat *every* point in the function's entire domain. Every global minimum is automatically a local minimum, but the converse is false — a function may have many local minima of which only the smallest is global.

### Geometric Interpretation

**From the Lecture — two supporting diagrams:**

1. A curve with peaks $A_1, A_2, A_3$ and valleys $B_1, B_2$ between endpoints $a$ and $b$. $B_1$ and $B_2$ are both **relative (local) minima**; $B_1$, being the lower of the two, is additionally the **global minimum** on $[a,b]$.
2. A single bowl-shaped ("convex-looking") curve between $a$ and $b$, where the **relative minimum is also the global minimum** — because there is only one "valley" in the whole domain, local and global minimality coincide.

```text
     Diagram 1: multiple local minima              Diagram 2: single (global) minimum
        A2                                                
      A1  A3                                         
    _/  \/  \_                                       \        /
   /    B2    \                                       \      /
  /  B1        \                                       \____/
 a______________b                                     a        b
 B1,B2 = relative minima; B1 = global minimum          the only relative min IS the global min
```

**Additional Explanation:** Diagram 2 is precisely the shape produced by a **convex function** (formally introduced later in the course, but foreshadowed in the lecture's Convex Optimization classification, Concept 14). This is why convexity matters so much in optimization: it *guarantees* Diagram 2's situation rather than Diagram 1's, eliminating the risk of getting "trapped" at an inferior local minimum.

---

## Concept 8 — Necessary Condition for a Single-Variable Local Extremum

### Statement (From the Lecture)

> **Theorem (Necessary condition).** If $f(x)$ is defined on $a \le x \le b$ and has a local (relative) minimum at $x=x^*$, where $a<x^*<b$, and if $f'(x^*)$ exists, then $f'(x^*)=0$.

### Full Proof (From the Lecture, Explained Step-by-Step)

**Step 1 — Set up the derivative as a limit.** By definition,
$$
f'(x^*) = \lim_{h\to 0} \frac{f(x^*+h)-f(x^*)}{h}
$$
and we are *given* that this limit exists (that is the hypothesis "$f'(x^*)$ exists"). The goal of the proof is to show that this limit must equal exactly zero.

**Step 2 — Use the local-minimum property.** Because $x^*$ is a local minimum, we know
$$
f(x^*) \le f(x^*+h)
$$
for every $h$ sufficiently close to $0$ (this is exactly the definition from Concept 7). Rearranged, this says the numerator $f(x^*+h)-f(x^*) \ge 0$ for all small $h$, whether $h$ is positive or negative.

**Step 3 — Take the one-sided limit as $h \to 0^+$ (from the right).** For $h>0$, dividing a non-negative numerator by a *positive* $h$ keeps the sign non-negative:
$$
\frac{f(x^*+h)-f(x^*)}{h} \ge 0 \quad (h>0) \implies \lim_{h\to 0^+}(\cdot) \ge 0 \implies f'(x^*) \ge 0
$$

**Step 4 — Take the one-sided limit as $h \to 0^-$ (from the left).** For $h<0$, dividing the same non-negative numerator by a *negative* $h$ flips the sign to non-positive:
$$
\frac{f(x^*+h)-f(x^*)}{h} \le 0 \quad (h<0) \implies \lim_{h\to 0^-}(\cdot) \le 0 \implies f'(x^*) \le 0
$$

**Step 5 — Combine.** Since the two-sided limit $f'(x^*)$ exists by hypothesis, both one-sided limits must equal it. Step 3 forces $f'(x^*) \ge 0$ and Step 4 forces $f'(x^*) \le 0$. The only number satisfying both $\ge 0$ **and** $\le 0$ simultaneously is $0$. Hence:
$$
f'(x^*) = 0 \qquad \blacksquare
$$

**Additional Explanation — why this argument is elegant:** The proof doesn't use any calculus "tricks" beyond the definition of the derivative and simple sign reasoning. It shows the necessary condition is really just a consequence of *consistency*: if the slope "from the left" and the slope "from the right" of the minimum point are forced to have opposite signs (or be zero) by the minimality property, and the two-sided derivative exists at all, then those slopes must agree — and the only value they can agree on is zero.

### Notes / Limitations (From the Lecture, Explained)

1. **The same conclusion holds for a local maximum** (the lecture leaves this as a "try it yourself" exercise). **Additional Explanation:** the proof is symmetric — flip the local-minimum inequality to $f(x^*) \ge f(x^*+h)$, and every $\ge$/$\le$ in Steps 3–4 flips accordingly, still forcing $f'(x^*)=0$ in the end.
2. **The theorem does not apply if $f'(x^*)$ does not exist.** Example: $f(x)=|x|$ has a global (and local) minimum at $x=0$, but $f'(0)$ does not exist (the left-hand slope is $-1$, the right-hand slope is $+1$, so no two-sided derivative exists) — the theorem's *hypothesis* is violated, so it says nothing here, even though $x=0$ genuinely is the minimum.
3. **The theorem does not apply at endpoints.** At $x=a$ or $x=b$, only a *one-sided* derivative can be defined (you cannot approach from both directions while staying in $[a,b]$), so the two-sided-limit argument of Steps 3–5 above cannot be completed — an endpoint minimum need not have zero one-sided derivative.
4. **The converse is false: $f'(x^*)=0$ does NOT imply $x^*$ is a local min or max.** Example: $f(x)=x^3$ has $f'(0)=0$, but $x=0$ is neither a local minimum nor a local maximum of $f$ — it is an **inflection point** (the function is momentarily "flat" but continues increasing on both sides).

### Common Misconceptions

- **Misconception:** "If $f'(x^*)=0$, then $x^*$ must be a local min or max." **Correction:** False — see Note 4 above ($f(x)=x^3$). A point where $f'(x^*)=0$ is only called a **stationary point**; it requires a *further* test (the sufficient condition, Concept 10 below) to classify it as min, max, or neither.
- **Misconception:** "The necessary condition always applies." **Correction:** It requires the derivative to exist and the point to be interior (not an endpoint) — see Notes 2 and 3.

---

## Concept 9 — Worked Example: Classifying Stationary Points of a Quintic

### Problem (From the Lecture)

Determine the max and min values of
$$
f(x) = 12x^5 - 45x^4 + 40x^3 + 5
$$

### Solution (From the Lecture, With Every Calculation Shown)

**From the Lecture, the stated results:**
- Stationary points: $f'(x)=0 \Rightarrow x=0,1,2$.
- At $x=2$: minimum, since $f''(2)>0$, with $f_{\min}=-11$.
- At $x=1$: maximum, since $f''(1)<0$, with $f_{\max}=12$.
- At $x=0$: inflection point, since $f''(0)=0$ and $f'''(0)\ne 0$.

**Additional Explanation — the lecture states these results but does not show the underlying calculus. Here is the full derivation:**

**Step 1 — Compute the first derivative.**
$$
f'(x) = 60x^4 - 180x^3 + 120x^2
$$

**Step 2 — Factor to find the stationary points.**
$$
f'(x) = 60x^2(x^2 - 3x + 2) = 60x^2(x-1)(x-2)
$$
Setting $f'(x)=0$ gives $x=0$ (a double root), $x=1$, and $x=2$ — exactly matching the lecture's stated stationary points.

**Step 3 — Compute the second derivative.**
$$
f''(x) = 240x^3 - 540x^2 + 240x = 60x(4x^2 - 9x + 4)
$$

**Step 4 — Evaluate $f''$ at each stationary point.**
- $f''(1) = 60(1)(4-9+4) = 60(-1) = -60 < 0 \implies$ local **maximum** at $x=1$ (matches the sufficient-condition theorem of Concept 10: $n=2$, even, $f^{(2)}(x^*)<0$).
- $f''(2) = 60(2)(16-18+4) = 120(2) = 240 > 0 \implies$ local **minimum** at $x=2$ ($n=2$, even, $f^{(2)}(x^*)>0$).
- $f''(0) = 0$, so the second derivative test is inconclusive at $x=0$; we must go to the next derivative.

**Step 5 — Compute the third derivative and evaluate at $x=0$.**
$$
f'''(x) = 720x^2 - 1080x + 240 \implies f'''(0) = 240 \ne 0
$$
Since the first *nonzero* higher derivative at $x=0$ is $f'''(0)$ (order $n=3$, which is **odd**), the sufficient-condition theorem (Concept 10) says $x=0$ is **neither a max nor a min** — it is an inflection point, exactly as the lecture states.

**Step 6 — Evaluate the function at the extrema.**
$$
f(1) = 12(1)-45(1)+40(1)+5 = 12-45+40+5 = 12 \quad (\text{matches } f_{\max}=12)
$$
$$
f(2) = 12(32) - 45(16) + 40(8) + 5 = 384-720+320+5 = -11 \quad (\text{matches } f_{\min}=-11)
$$

**Diagram (From the Lecture):** A plot of $f(x)$ over roughly $x\in[-0.2, 2.2]$, $y\in[-15,15]$: the curve rises from about $(0,5)$, flattens momentarily near $x=0$ (the inflection point — notice it is *not* a peak or valley, just a "shoulder"), continues rising to a peak near $(1,12)$, descends through zero, reaches a valley near $(2,-11)$, then rises sharply again past $x=2$.

**Additional Explanation — why this example is pedagogically important:** It is a single, self-contained example that exercises *every* case of the sufficient-condition theorem at once: an even-order derivative test giving a max, an even-order derivative test giving a min, and an odd-order derivative test giving neither. This is precisely why the instructor chose a quintic (degree-5) polynomial rather than something simpler like a quadratic (which can only ever show one case).

---

## Concept 10 — Sufficient Condition for a Single-Variable Local Extremum

### Statement (From the Lecture)

> **Theorem (Sufficient Condition).** Let $f'(x^*) = \dots = f^{(n-1)}(x^*) = 0$, but $f^{(n)}(x^*) \ne 0$. Then $f(x^*)$ is:
> - a **minimum** value of $f$ if $f^{(n)}(x^*) > 0$ and $n$ is **even**.
> - a **maximum** value of $f$ if $f^{(n)}(x^*) < 0$ and $n$ is **even**.
> - **neither** a max nor a min if $n$ is **odd**.

### Proof (From the Lecture, as far as given)

**From the Lecture:** By applying Taylor's theorem,
$$
f(x^*+h) - f(x^*) = \frac{h^n}{n!} f^{(n)}(x^* + \theta h)
$$

> **[UNCLEAR IN SOURCE — DO NOT INFER: the lecture's proof stops immediately after stating this Taylor-expansion identity. No further algebraic steps, sign analysis, or concluding statement are provided in the source slides.]**

**Additional Explanation — completing the reasoning the lecture leaves implicit:**

This equation is the **Taylor expansion of $f$ around $x^*$ with the Lagrange form of the remainder**, using the fact that *all* derivative orders below $n$ vanish at $x^*$ (so every term of the expansion up to order $n-1$ disappears, leaving only the order-$n$ term, evaluated not exactly at $x^*$ but at some intermediate point $x^*+\theta h$ for some $\theta \in (0,1)$ — this is the "Mean Value" flavor of Taylor's theorem).

To finish the proof, reason about the **sign** of the right-hand side, $\frac{h^n}{n!}f^{(n)}(x^*+\theta h)$, for $h$ small:

- Because $f^{(n)}$ is continuous and $f^{(n)}(x^*)\ne 0$, for $h$ small enough $f^{(n)}(x^*+\theta h)$ has the **same sign** as $f^{(n)}(x^*)$ (continuity — a continuous nonzero function keeps its sign in a small enough neighborhood).
- **If $n$ is even:** $h^n \ge 0$ for *every* real $h$ (positive or negative), since an even power of any real number is non-negative. So the sign of the whole right-hand side is controlled *only* by the sign of $f^{(n)}(x^*)$:
  - If $f^{(n)}(x^*) > 0$: the right-hand side is $\ge 0$ for all small $h$, i.e., $f(x^*+h) \ge f(x^*)$ for all small $h$ — this is precisely the definition of a **local minimum**.
  - If $f^{(n)}(x^*) < 0$: the right-hand side is $\le 0$ for all small $h$, i.e., $f(x^*+h) \le f(x^*)$ — a **local maximum**.
- **If $n$ is odd:** $h^n$ **changes sign** depending on whether $h>0$ or $h<0$ (e.g., $h^3>0$ for $h>0$ but $h^3<0$ for $h<0$). So the right-hand side is positive on one side of $x^*$ and negative on the other, meaning $f(x^*+h)-f(x^*)$ changes sign as $h$ crosses zero — $f$ is *larger* than $f(x^*)$ on one side and *smaller* on the other. This means $x^*$ is **neither** a local min nor a local max — it is an inflection point (exactly what happened at $x=0$ in Concept 9's worked example, where $n=3$).

### Intuition

**Additional Explanation:** This theorem generalizes the familiar "second derivative test" from introductory calculus ($f''(x^*)>0 \Rightarrow$ min, $f''(x^*)<0\Rightarrow$ max) to the case where the second derivative is *also* zero. It tells you: keep differentiating until you hit a nonzero derivative; the *parity* (even/odd) of that derivative's order tells you whether the curve behaves symmetrically around $x^*$ (even — a genuine extremum) or antisymmetrically (odd — an inflection/saddle-like flat spot).

### AI / Machine Learning Connection

**Additional Explanation:** In practice, virtually all optimization algorithms used in ML (gradient descent, Newton's method, etc.) rely on the **first- and second-order conditions only** ($n=1$ and $n=2$); they almost never compute third- or higher-order derivatives, both because higher derivatives are expensive to compute for high-dimensional models and because, generically, a random function's Hessian is non-degenerate (so the $n=2$ test almost always suffices in practice — degenerate cases like $x^3$ are measure-zero/non-generic).

---

## Concept 11 — Multivariable Optimization: Necessary Condition

### Statement (From the Lecture)

> **Theorem (Necessary Condition).** If $f(\mathbf{x})$ has a local min (max) at $\mathbf{x}=\mathbf{x}^*$, and if the first partial derivatives of $f(\mathbf{x})$ exist at $\mathbf{x}^*$, then $\nabla f(\mathbf{x}^*) = 0$.

### The Gradient — Mathematical Meaning

**Additional Explanation (prerequisite material needed to use this theorem):** The gradient $\nabla f(\mathbf{x})$ of a function $f:\mathbb{R}^n \to \mathbb{R}$ is the vector of all first partial derivatives:
$$
\nabla f(\mathbf{x}) = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right)
$$
It has the same dimension $n$ as the decision variable $\mathbf{x}$. Geometrically, $\nabla f(\mathbf{x})$ points in the direction of **steepest ascent** of $f$ at $\mathbf{x}$ (the direction in which $f$ increases fastest), and its magnitude tells you how fast $f$ is increasing in that direction. The negative gradient $-\nabla f(\mathbf{x})$ therefore points in the direction of **steepest descent** — this is the entire basis of the gradient descent algorithm used to train ML models. The gradient is also always **perpendicular (normal) to the level set/contour** of $f$ passing through $\mathbf{x}$.

### Proof (From the Lecture, Explained Step-by-Step)

**Step 1 — Write the multivariable first-order Taylor expansion.**
$$
f(\mathbf{x}^*+\mathbf{h}) - f(\mathbf{x}^*) = \mathbf{h}^T \nabla f(\mathbf{x}^*) + R_1(\mathbf{x}^*, \mathbf{h})
$$
where $\mathbf{h}^T \nabla f(\mathbf{x}^*)$ is the dot product between the perturbation direction $\mathbf{h}$ and the gradient (this is the multivariable analogue of the single-variable first-order term $h\, f'(x^*)$), and $R_1$ collects all higher-order (quadratic and beyond) terms.

**Step 2 — Argue about dominance of the linear term for small $\mathbf{h}$.**
**From the Lecture:** "The 1st order terms of $\mathbf{h}$ dominate the higher orders for small $\mathbf{h}$." **Additional Explanation:** As $\|\mathbf{h}\| \to 0$, the remainder $R_1(\mathbf{x}^*,\mathbf{h})$ shrinks *faster* than $\mathbf{h}^T \nabla f(\mathbf{x}^*)$ (formally, $R_1 = o(\|\mathbf{h}\|)$, i.e. it vanishes faster than linearly), so for sufficiently small $\mathbf{h}$, the **sign** of the left-hand side $f(\mathbf{x}^*+\mathbf{h})-f(\mathbf{x}^*)$ is controlled entirely by the sign of the linear term $\mathbf{h}^T\nabla f(\mathbf{x}^*)$, exactly as in the lecture's statement: "The sign of LHS depends on the sign of 1st term (component-wise)."

**Step 3 — Argue by contradiction.** Suppose $\nabla f(\mathbf{x}^*) \ne 0$. Then you can choose a direction $\mathbf{h}$ such that $\mathbf{h}^T\nabla f(\mathbf{x}^*) < 0$ (simply pick $\mathbf{h}$ pointing opposite to $\nabla f(\mathbf{x}^*)$, i.e. $\mathbf{h}=-\epsilon\,\nabla f(\mathbf{x}^*)$ for small $\epsilon>0$; then $\mathbf{h}^T\nabla f(\mathbf{x}^*) = -\epsilon\|\nabla f(\mathbf{x}^*)\|^2 < 0$). By Step 2, this would make $f(\mathbf{x}^*+\mathbf{h}) - f(\mathbf{x}^*) < 0$ for small enough $\mathbf{h}$ — meaning you could *decrease* $f$ below $f(\mathbf{x}^*)$ by moving in this direction, contradicting the assumption that $\mathbf{x}^*$ is a local minimum. (You could also pick the *opposite* direction $\mathbf{h}=+\epsilon\nabla f(\mathbf{x}^*)$ to get $f(\mathbf{x}^*+\mathbf{h}) > f(\mathbf{x}^*)$, contradicting a local max, if that is instead the assumption.)

**Step 4 — Conclude.** Since assuming $\nabla f(\mathbf{x}^*)\ne 0$ leads to a contradiction (we found a direction that strictly *decreases* $f$, so $\mathbf{x}^*$ could not have been a local min after all), it must be that:
$$
\nabla f(\mathbf{x}^*) = 0 \qquad \blacksquare
$$

**Additional Explanation:** This is the direct multivariable generalization of Concept 8's proof: instead of using only two directions ($h>0$ and $h<0$ along a single axis), the multivariable proof shows that *every possible direction* $\mathbf{h}$ must fail to strictly improve $f$ near a local minimum — and the only vector $\nabla f(\mathbf{x}^*)$ for which *no* direction can produce a strict decrease is the zero vector.

### Geometric Interpretation

At an unconstrained local minimum, the "tangent plane" to the graph of $f$ is horizontal in every direction simultaneously — there is no "uphill/downhill" slope left in any direction, which is exactly what $\nabla f(\mathbf{x}^*)=0$ encodes.

---

## Concept 12 — Multivariable Optimization: Sufficient Condition (The Hessian Test)

### The Hessian Matrix — Mathematical Meaning

**Additional Explanation (prerequisite for the theorem):** The Hessian matrix $H(\mathbf{x}) = \nabla^2 f(\mathbf{x})$ of $f:\mathbb{R}^n\to\mathbb{R}$ is the $n\times n$ matrix of all second partial derivatives:
$$
H(\mathbf{x})_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$
It is symmetric (assuming $f$ is twice continuously differentiable, by Clairaut's/Schwarz's theorem, $\partial^2 f/\partial x_i \partial x_j = \partial^2 f/\partial x_j \partial x_i$). It encodes **curvature**: how the gradient itself is changing, i.e., whether the function is bending "upward" (locally convex-like, bowl-shaped) or "downward" (locally concave-like) in each direction and combination of directions.

### Statement (From the Lecture)

> **Theorem (Sufficient Condition).** A sufficient condition for a stationary point $\mathbf{x}^*$ to be a local min is that the Hessian matrix $H(\mathbf{x}^*)=\nabla^2 f(\mathbf{x}^*)$ is **positive definite (PD)**: $H(\mathbf{x}^*) \succ 0$.

### Proof (From the Lecture, Explained Step-by-Step)

**Step 1 — Write the second-order multivariable Taylor expansion.**
$$
f(\mathbf{x}^*+\mathbf{h}) - f(\mathbf{x}^*) = \mathbf{h}^T \nabla f(\mathbf{x}^*) + \frac{1}{2}\mathbf{h}^T \nabla^2 f(\mathbf{x}^*+\theta\mathbf{h})\,\mathbf{h}, \qquad 0<\theta<1
$$
This is the multivariable analogue of the single-variable Taylor theorem with remainder used in Concept 10, now carried out to second order, with the quadratic term evaluated at an intermediate point $\mathbf{x}^*+\theta\mathbf{h}$.

**Step 2 — Apply the fact that $\mathbf{x}^*$ is a stationary point.** By Concept 11, at a stationary point $\nabla f(\mathbf{x}^*) = 0$, so the first-order term vanishes entirely:
$$
f(\mathbf{x}^*+\mathbf{h}) - f(\mathbf{x}^*) = \frac{1}{2}\mathbf{h}^T \nabla^2 f(\mathbf{x}^*+\theta\mathbf{h})\,\mathbf{h}, \qquad 0<\theta<1
$$

**Step 3 — Require the right-hand side to be positive.** **From the Lecture:** "To be a local min, the RHS should be $>0$." **Additional Explanation:** if we can guarantee $\frac{1}{2}\mathbf{h}^T\nabla^2f(\mathbf{x}^*+\theta\mathbf{h})\mathbf{h} > 0$ for every small nonzero $\mathbf{h}$, then $f(\mathbf{x}^*+\mathbf{h}) > f(\mathbf{x}^*)$ for every small perturbation — precisely the definition of a strict local minimum. By continuity of $\nabla^2 f$, if $H(\mathbf{x}^*) = \nabla^2 f(\mathbf{x}^*)$ is positive definite, then $\nabla^2 f(\mathbf{x}^*+\theta\mathbf{h})$ is *also* positive definite for $\mathbf{h}$ small enough (positive definiteness is an open condition preserved under small perturbations), guaranteeing the needed sign.

**Step 4 — Conclude.**
$$
H(\mathbf{x}^*) = \nabla^2 f(\mathbf{x}^*) \succ 0 \qquad \blacksquare
$$

### Positive Definite Matrices — Checkable Conditions (From the Lecture)

**From the Lecture,** a symmetric matrix $A$ is **positive definite (PD)** if **any one** (equivalently, all) of the following hold:
- Every eigenvalue of $A$ is positive.
- Every (leading principal) minor determinant of $A$ is positive.
- $\mathbf{h}^T A \mathbf{h} > 0$ for **all** $\mathbf{h}\ne 0$.

**Additional Explanation — why these three tests are equivalent, and how to use them:**
- The **quadratic-form test** ($\mathbf{h}^T A \mathbf{h}>0\ \forall \mathbf{h}\ne 0$) is the *definition* of positive definiteness; the other two are practical, checkable criteria that are mathematically guaranteed to give the same answer.
- The **eigenvalue test** is conceptually the clearest: a symmetric matrix can always be diagonalized in an orthonormal eigenbasis, and in that basis $\mathbf{h}^TA\mathbf{h} = \sum_k \lambda_k y_k^2$ (where $y_k$ are the coordinates of $\mathbf{h}$ in the eigenbasis and $\lambda_k$ the eigenvalues) — this sum is guaranteed positive for every nonzero $\mathbf{h}$ exactly when every $\lambda_k>0$.
- The **leading principal minors test** (also known as Sylvester's criterion) is the most practical "by hand" test for small matrices: compute the determinants of the top-left $1\times1$, $2\times2$, ..., $n\times n$ submatrices; if *all* are strictly positive, $A$ is PD.

> **From the Lecture — reflection prompt:** *"Think about negative definite matrix!"*
>
> **[UNCLEAR IN SOURCE — DO NOT INFER: no worked answer is provided.]**
>
> **Additional Explanation (answering the spirit of the prompt):** By symmetry with the PD case (and because $A$ negative definite $\iff$ $-A$ positive definite), a symmetric matrix $A$ is **negative definite (ND)** if: every eigenvalue is negative; the leading principal minors **alternate in sign** starting negative ($\det(A_1)<0, \det(A_2)>0, \det(A_3)<0,\dots$); or $\mathbf{h}^TA\mathbf{h}<0$ for all $\mathbf{h}\ne 0$. A negative definite Hessian at a stationary point is the multivariable sufficient condition for a **local maximum**, exactly mirroring the min/max symmetry seen already in the single-variable theorem of Concept 10.

### Example 2, Revisited — Sanity-Checking with a Bordered Hessian

**Additional Explanation (Example, added to connect Concept 12 back to Example 2):** In Example 2 (maximum-area rectangle), after eliminating $h$ via the constraint, the single-variable function $A(w) = \frac{L}{2}w - w^2$ has $A''(w)=-2$ for all $w$ — a "$1\times1$ negative-definite Hessian" (trivially, since it's a scalar), consistent with the result that $w=L/4$ is a **maximum**, matching the ND discussion just above.

---

## Concept 13 — The Semi-Definite Case and Saddle Points

### The Semi-Definite Case (From the Lecture)

$$
H(\mathbf{x}^*) = \nabla^2 f(\mathbf{x}^*) \succeq 0
$$

**From the Lecture:** when the Hessian is only positive **semi**-definite (i.e., $\mathbf{h}^TA\mathbf{h}\ge 0$ for all $\mathbf{h}$, but possibly $=0$ for some nonzero $\mathbf{h}$) rather than strictly positive definite, the sufficient-condition test of Concept 12 is inconclusive, and you must **investigate the higher-order derivatives in the Taylor series expansion** — this is the direct multivariable analogue of Concept 10's approach for single-variable functions, where a zero second derivative forced you to check the third, fourth, etc.

### Saddle Points (From the Lecture)

**From the Lecture:**
- In the case of a function of **two** variables, the Hessian matrix may be **neither positive nor negative definite** at $(x^*,y^*)$ — this situation defines a **saddle point**.
- Saddle points may also exist for functions of **more than two** variables.

**Example (From the Lecture):**
$$
f(x,y) = x^2 - y^2
$$

**Diagram (From the Lecture, Figure 2.5):** a 3D surface that curves *upward* along the $x$-direction and *downward* along the $y$-direction, resembling a horse's saddle or a mountain pass — it has a stationary point at the origin that is a minimum "if you only look along $x$" but a maximum "if you only look along $y$" simultaneously.

**Additional Explanation — verifying this is a saddle point using the tools above:**
The gradient is $\nabla f(x,y) = (2x, -2y)$, which vanishes only at $(0,0)$ — so $(0,0)$ is the unique stationary point. The Hessian is
$$
H = \begin{pmatrix} 2 & 0 \\ 0 & -2 \end{pmatrix}
$$
This matrix's eigenvalues are $2$ and $-2$ — one positive, one negative. By the eigenvalue test above, $H$ is neither PD ($-2<0$ violates "all eigenvalues positive") nor ND ($2>0$ violates "all eigenvalues negative"); it is **indefinite**. This is the textbook signature of a saddle point: a Hessian with *both* positive and negative eigenvalues, meaning the function curves upward in some directions and downward in others, simultaneously, at the same stationary point.

**Exercise (From the Lecture, posed without a worked solution):**

> Find the extreme points for the function: $f(x_1,x_2) = x_1^3 + x_2^3 + 2x_1^2 + 4x_2^2 + 6$.

> **[UNCLEAR IN SOURCE — DO NOT INFER: this exercise is presented in the lecture without a worked solution.]**

**Additional Explanation (worked as a self-check, clearly separated from the source lecture):**

*Step 1 — Gradient.*
$$
\nabla f = \left( 3x_1^2+4x_1,\ 3x_2^2+8x_2 \right)
$$

*Step 2 — Set to zero and solve.*
$$
3x_1^2+4x_1 = x_1(3x_1+4) = 0 \implies x_1 = 0 \text{ or } x_1 = -\tfrac{4}{3}
$$
$$
3x_2^2+8x_2 = x_2(3x_2+8) = 0 \implies x_2 = 0 \text{ or } x_2 = -\tfrac{8}{3}
$$
This gives **four** candidate stationary points: $(0,0)$, $(0,-\tfrac83)$, $(-\tfrac43,0)$, $(-\tfrac43,-\tfrac83)$.

*Step 3 — Hessian.*
$$
H(x_1,x_2) = \begin{pmatrix} 6x_1+4 & 0 \\ 0 & 6x_2+8 \end{pmatrix}
$$
(diagonal, because $f$ separates into a function of $x_1$ alone plus a function of $x_2$ alone).

*Step 4 — Classify each candidate:*

| Point | $6x_1+4$ | $6x_2+8$ | Classification |
|---|---|---|---|
| $(0,0)$ | $4>0$ | $8>0$ | Both eigenvalues $>0$ → **PD → local min** |
| $(0,-\tfrac83)$ | $4>0$ | $-8<0$ | mixed signs → **indefinite → saddle point** |
| $(-\tfrac43,0)$ | $-4<0$ | $8>0$ | mixed signs → **indefinite → saddle point** |
| $(-\tfrac43,-\tfrac83)$ | $-4<0$ | $-8<0$ | Both eigenvalues $<0$ → **ND → local max** |

This worked exercise is a clean, added illustration of how the necessary condition (Concept 11) generates *candidate* stationary points, and the sufficient condition (Concept 12/13) then classifies each one individually — different stationary points of the *same* function can have entirely different classifications.

### Additional Multivariable Visualizations (From the Lecture, Descriptive)

The lecture includes several supporting figures illustrating these ideas visually, without further accompanying text beyond section headers:

1. Two 3D surface plots of undulating multi-basin functions $f(x_1,x_2)$, illustrating multiple local extrema on a single surface.
2. Two contour plots with labeled level values, each showing **two distinct closed-contour basin regions** — visual confirmation of multiple local optima.
3. A fine-grained contour plot showing two basin structures with contour lines **pinching together** near the center — the geometric signature of a saddle point between two basins (contours of a saddle point look like hyperbolas rather than closed loops, exactly as in the $f(x,y)=x^2-y^2$ example above).
4. A real-world topographic (elevation) map of Yosemite National Park — used as a physical analogy: elevation contour lines on a hiking map are literally level curves of the "elevation" function, and peaks/valleys/passes on the map correspond exactly to local maxima, local minima, and saddle points of that function.
5. A geographic contour/heatmap plot resembling a regional map, showing colored contour bands of a spatially varying quantity.
6. A meteorological streamline plot ("Streamlines (700mb)") over the continental United States, showing wind-flow patterns with visible vortices — analogous to gradient/flow fields, where streamlines are everywhere tangent to a vector field (conceptually related to gradient-flow trajectories used in optimization algorithms like gradient descent, which move along $-\nabla f$).

**Additional Explanation:** These figures collectively reinforce a single idea: **contour plots and 3D surface plots are visual proxies for the algebraic classification (PD/ND/indefinite Hessian) developed above.** Closed, nested contours around a point signal a local extremum (min or max); contours that cross or pinch into an "X" or hyperbola shape signal a saddle point.

---

## Concept 14 — Multivariable Optimization with Equality Constraints (Named Methods)

**From the Lecture:**
$$
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_j(\mathbf{x}) = 0, \quad j\in J=\{1,\dots,m\}
$$

Three solution approaches are **named** (but not elaborated in this week's slides):
- Solution with **direct substitution**
- Solution by the **method of constrained variation**
- Solution by the **method of Lagrange Multipliers**

> **[UNCLEAR/INCOMPLETE — the lecture lists these three method names as section headings for this topic but provides no further mathematical development within these Week 1 slides; presumably these are developed in a subsequent lecture.]**

**Additional Explanation (a brief preview, clearly separated from the lecture, to connect this heading to material you already used above):** "Direct substitution" is exactly the technique you already used, in full, to solve **Example 2** (maximum-area rectangle) above: you used the equality constraint $L=2(w+h)$ to eliminate one variable, then applied unconstrained single-variable optimization to what remained. This works well when a constraint can be solved explicitly for one variable in terms of the others; when it cannot (or when there are several constraints, making substitution unwieldy), the method of Lagrange multipliers is the general-purpose tool developed later in the course. Since the KKT conditions (mentioned below) are the *inequality-constrained* generalization of the Lagrange-multiplier idea, mastering the concepts already proven in this chapter (gradients, Hessians, stationary points) is the direct prerequisite for that later material.

---

## Concept 15 — Multivariable Optimization with Inequality Constraints (Named Method)

**From the Lecture:**
$$
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_j(\mathbf{x}) \le 0, \quad j\in J=\{1,\dots,m\}
$$

The named solution approach: **Kuhn–Tucker (KKT) conditions**.

> **[UNCLEAR/INCOMPLETE — the lecture names KKT conditions as the method for this section but does not present the conditions themselves within these Week 1 slides; presumably developed in a subsequent lecture.]**

**Additional Explanation:** The active/inactive constraint distinction you learned in Concept 6 is the essential prerequisite idea for KKT: intuitively, the KKT conditions say that at an optimal point, the negative gradient of the objective must be expressible as a non-negative combination of the gradients of the *active* constraints only (inactive constraints, having no local effect, contribute nothing). This is a direct generalization of the Lagrange-multiplier idea (Concept 14) from equality constraints to inequality constraints.

---

## Concept 16 — Classification of Optimization Problems

The lecture classifies optimization problems into several families based on the mathematical structure of $f$, $g_j$, and $h_i$. Each is presented below with its definition, why it matters, and, where the lecture connects it to a worked example, that connection.

### 16.1 Linear Programming (LP)

**Definition (From the Lecture):** If all the functions $f, h_1,\dots,h_p,g_1,\dots,g_m$ are **affine (linear)** functions of $\mathbf{x}$:
$$
\min_{\mathbf{x}} \mathbf{C}^T\mathbf{x} \quad \text{s.t.}\quad \mathbf{A}\mathbf{x} = \mathbf{b}
$$

**Mathematical Meaning:** $\mathbf{C}\in\mathbb{R}^n$ is a fixed cost-coefficient vector, $\mathbf{C}^T\mathbf{x}$ is a linear (weighted-sum) objective, $\mathbf{A}$ is an $m\times n$ matrix of constraint coefficients, and $\mathbf{b}\in\mathbb{R}^m$ is the right-hand-side vector of resource limits — the entire problem is described only by matrices and vectors, no curved/nonlinear terms anywhere.

**Additional Explanation:** An **affine** function is one of the form $c_0 + \mathbf{c}^T\mathbf{x}$ (a linear function plus a constant); "linear programming" technically permits this constant offset even though it is called "linear." The half of the course title, "Linear... Programming," refers to this exact class.

### 16.2 Unconstrained Optimization

**Definition (From the Lecture):** The index sets $I,J$ are empty, and $\mathbf{x}\in\mathbb{R}^n$: $\min_{\mathbf{x}} f(\mathbf{x})$.

**Additional Explanation:** This is the setting in which the necessary/sufficient conditions of Concepts 11–13 apply directly and completely, with no need for Lagrange multipliers or KKT conditions. Neural network training (Example 3(a)) is, in its purest textbook form, an unconstrained nonlinear optimization problem.

### 16.3 Convex Optimization

**Definition (From the Lecture):** $f, g_1,\dots,g_m$ are **convex functions**, and $h_1,\dots,h_p$ are **affine**, and $\text{dom}(f)$ is convex.

**Additional Explanation — why this classification is so important (prerequisite concept, not detailed further by the lecture itself but essential background):** A function $f$ is convex if, for every two points $\mathbf{x},\mathbf{y}$ in its domain and every $\lambda\in[0,1]$,
$$
f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \le \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y})
$$
i.e., the graph of $f$ never lies above the straight line ("chord") connecting any two of its points. The single most important consequence — foreshadowed already by Concept 7's Diagram 2 ("the only relative min IS the global min") — is that **for a convex optimization problem, every local minimum is automatically a global minimum.** This is precisely why convex optimization is prized in engineering and machine learning: the theory developed in this chapter (find a stationary point, check the Hessian is PD) is not just *locally* useful — for a convex problem it certifies *global* optimality.

### 16.4 Quadratic Optimization

**Definition (From the Lecture):** The objective $f$ is quadratic, and $h_1,\dots,h_p,g_1,\dots,g_m$ are affine. The quadratic objective is given by:
$$
f(\mathbf{x}) = a + \mathbf{b}^T\mathbf{x} + \mathbf{x}^T H \mathbf{x}
$$

**Mathematical Meaning:** $a\in\mathbb{R}$ is a constant term, $\mathbf{b}\in\mathbb{R}^n$ is a linear-coefficient vector, and $H$ is (typically) a symmetric $n\times n$ matrix. Notice this is the exact structure of a second-order Taylor expansion around a point (compare to the Taylor expansion in Concept 12's proof!) — this is not a coincidence: quadratic optimization is, in a precise sense, the *local model* that the sufficient-condition analysis (Concept 12) uses to approximate any smooth nonlinear $f$ near a candidate stationary point.

**AI / Machine Learning Connection:** By the lecture's own Example 3(b), the SVM optimal-margin-hyperplane problem is a quadratic optimization problem (quadratic objective related to $\|\mathbf{w}\|^2$, i.e., inverse squared margin, subject to affine/linear classification-margin constraints).

### 16.5 Geometric Programming (GP)

**Definition (From the Lecture):**
$$
\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_j(\mathbf{x}) \le 1,\ j\in J=\{1,\dots,m\}, \quad \mathbf{x} > 0
$$
where $f,g_1,\dots,g_m$ are **posynomial** functions.

**Definition — posynomial function (From the Lecture):**
$$
f(\mathbf{x}) = \sum_{k=1}^{K} c_k\, x_1^{r_{1k}}x_2^{r_{2k}}\cdots x_n^{r_{nk}}, \qquad c_k \ge 0,\ r_{ij}\in\mathbb{R}
$$

**Example (From the Lecture):**
$$
f(x_1,x_2,x_3) = x_1^2 x_2^{-1}x_3^{1.2} + 2x_2^3x_3^{0.5}
$$

**Mathematical Meaning:** Each term of a posynomial is a **monomial** — a positive coefficient $c_k$ multiplying variables raised to *arbitrary real* (not necessarily integer, and possibly negative) powers $r_{ij}$ — summed together, with the entire sum required to have non-negative coefficients and to be evaluated only where every $x_i>0$. In the lecture's example, the two monomials are $x_1^2x_2^{-1}x_3^{1.2}$ and $2x_2^3x_3^{0.5}$; note the fractional exponent $1.2$ and the negative exponent $-1$, both of which are disallowed in ordinary polynomials but permitted here.

**Why It Exists (From the Lecture):** "GP arise in layout and circuits applications where $x_i$ are physical dimensions and constraints involve areas, weight and volumes" — this is precisely the structure of **Example 4** (device sizing): power, delay, and area in circuit design are frequently expressible as posynomials of the transistor widths/lengths. **Additional Explanation:** geometric programs have the remarkable property that a simple change of variables ($x_i = e^{y_i}$) converts them into **convex optimization problems** in the transformed variables $y_i$ — this is *why* GP is singled out as its own named class rather than being treated as "just another nonconvex problem": despite looking nonlinear and nonconvex in $\mathbf{x}$, GP is secretly convex (and therefore globally solvable) after a log-transform.

### 16.6 Discrete (Integer) Programming

**Definition (From the Lecture):** The solution is restricted to one of a number of objects in a **finite set** — mathematically, add the constraint
$$
x_i \in \mathbb{Z}, \quad i=1,\dots,n
$$
where $\mathbb{Z}$ is the set of all integers.

**From the Lecture:**
- One (imperfect) technique: solve with real (continuous) variables, then round every component to the nearest integer. **"No guarantee of optimal solution!"** — rounding a continuous-relaxation solution can produce a point that is no longer feasible, or is feasible but far from the true integer-constrained optimum.
- Exact/structured techniques named: **cutting plane method** and **branch-and-bound method**, applied to (integer) linear programming problems.

**Additional Explanation:** Example 1 (the airfare problem) is, informally, an integer/combinatorial optimization problem — you were choosing among a *discrete* set of travel plans rather than a continuous quantity, which is why the lecture solved it by direct enumeration (computing the cost of each of the three named alternatives) rather than by calculus.

### 16.7 Stochastic Programming

**Definition (From the Lecture):** In many economic/financial planning models — which often depend on future interest-rate movements and future economic behavior — the optimization model cannot be fully specified in advance (a **stochastic representation** is needed). Modelers predict/estimate unknown quantities with some degree of confidence, and stochastic optimization algorithms use these uncertainties to produce solutions that optimize the **expected performance** of the model.

**Additional Explanation:** The key mathematical shift here is that the objective is no longer $f(\mathbf{x})$ for a fixed, fully-known $f$, but rather $\mathbb{E}[f(\mathbf{x},\xi)]$, an *expectation* over a random variable $\xi$ (representing the unknown future data). This connects to ML in the form of **Stochastic Gradient Descent** (listed later in the lecture, Concept 17): rather than computing the exact gradient of a loss averaged over an entire dataset, SGD estimates it using a random mini-batch, trading exactness for computational speed — conceptually a stochastic-programming-flavored approach to optimization.

### 16.8 Multi-objective Optimization

**Definition (From the Lecture):** There may be more than one criterion to satisfy simultaneously — example given: a design for **min area and max SNR** (signal-to-noise ratio) in a communication system. With multiple objectives comes the possibility of *conflict*; one simple resolution is to build an overall objective as a linear combination of the individual objectives:
$$
f(\mathbf{x}) = \alpha_1 f_1(\mathbf{x}) + \alpha_2 f_2(\mathbf{x})
$$
where $\alpha_1,\alpha_2$ are constants indicating the relative importance of each objective. **Pareto front methods** are also named as an approach.

**Mathematical Meaning — why weights $\alpha_1,\alpha_2$ are needed:** since $f_1$ (area) and $f_2$ (negative SNR, say) are typically measured in different units and different scales, and improving one may worsen the other, there is generally no single point that simultaneously minimizes both. The weights let the designer express a *trade-off preference*. **Additional Explanation:** a **Pareto front** (mentioned but not elaborated in the lecture) is the set of all "non-dominated" solutions — points where you cannot improve one objective without worsening another; instead of picking one weighting $(\alpha_1,\alpha_2)$ in advance, Pareto-front methods characterize the entire trade-off curve and let the decision-maker choose a point on it after seeing the options.

### 16.9 Modern (Nontraditional/Heuristic) Optimization Methods

**Definition (From the Lecture):** Also called nontraditional optimization methods or heuristic techniques:
- ✔ Genetic Algorithms (GA)
- ✔ Simulated Annealing (SA)
- ✔ Particle Swarm Optimization (PSO)
- ✔ Ant Colony Optimization
- ✔ Artificial Neural Network (ANN)
- ✔ Fuzzy Logic optimization

**Additional Explanation:** Unlike LP, NLP, or the calculus-based methods of this chapter, these are typically **derivative-free, population-based, or nature-inspired** search strategies that do not require the objective function to be differentiable (or even known in closed form) — they instead rely on repeated sampling, comparison, and stochastic exploration of the search space. They are useful precisely where the classical necessary/sufficient conditions of Concepts 8–13 cannot be applied — e.g., when $f$ is non-differentiable, extremely non-convex with many local optima, or a "black box" (only evaluable, not analytically expressible).

### 16.10 Deep Learning Algorithms

**Definition (From the Lecture):**
- ✔ Gradient Descent
- ✔ Stochastic Gradient Descent
- ✔ Minibatch Stochastic Gradient Descent
- ✔ Momentum
- ✔ AdaGrad
- ✔ RMSProp
- ✔ ADAM

**Additional Explanation — direct connection back to Concepts 11–13:** Every algorithm in this list is, at its core, an iterative procedure for finding a point $\mathbf{x}^*$ (the network weights) satisfying (approximately) the necessary condition of Concept 11, $\nabla f(\mathbf{x}^*)\approx 0$, for a loss function $f$. Plain **Gradient Descent** takes repeated steps $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)$ (moving in the direction of steepest descent, $-\nabla f$, as defined in Concept 11's discussion of the gradient). **Stochastic** and **Minibatch** Gradient Descent replace the exact gradient with a noisy estimate computed from a random subset of the training data (connecting to the Stochastic Programming idea of Concept 16.7). **Momentum, AdaGrad, RMSProp, and ADAM** are all refinements that adapt the step size and/or direction using accumulated gradient history, in order to converge faster and more robustly than plain gradient descent — but none of them change the underlying target: driving the gradient toward zero at a (hopefully good) local minimum of the loss surface.

---

# Mathematical Foundations (Consolidated)

This section consolidates the calculus/linear-algebra machinery used throughout the chapter, for quick reference.

### The Gradient
$$
\nabla f(\mathbf{x}) = \left(\frac{\partial f}{\partial x_1},\dots,\frac{\partial f}{\partial x_n}\right)
$$
- **Contains:** all first partial derivatives.
- **Dimension:** same as $\mathbf{x}$, i.e., $n \times 1$.
- **Directional meaning:** points in the direction of steepest ascent; $-\nabla f$ points in the direction of steepest descent.
- **Level-set relationship:** always orthogonal (normal) to the level curve/surface of $f$ passing through $\mathbf{x}$.
- **Role in optimization:** the necessary condition for an unconstrained local extremum is $\nabla f(\mathbf{x}^*)=0$ (Concept 11).

### The Hessian
$$
\nabla^2 f(\mathbf{x})_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$
- **Contains:** all second partial derivatives.
- **Dimension:** $n\times n$, symmetric (for $C^2$ functions).
- **Meaning:** local curvature of $f$.
- **Definiteness and classification:**

| Hessian at stationary point | Classification |
|---|---|
| Positive definite ($\succ 0$) | Local minimum |
| Negative definite ($\prec 0$) | Local maximum |
| Indefinite (mixed-sign eigenvalues) | Saddle point |
| Positive/negative **semi**-definite (some eigenvalue $=0$) | Inconclusive — check higher-order derivatives |

### Taylor Expansions Used in This Chapter

- **Single variable, order $n$:** $f(x^*+h)-f(x^*) = \dfrac{h^n}{n!}f^{(n)}(x^*+\theta h)$, when $f'(x^*)=\dots=f^{(n-1)}(x^*)=0$.
- **Multivariable, first order:** $f(\mathbf{x}^*+\mathbf{h})-f(\mathbf{x}^*) = \mathbf{h}^T\nabla f(\mathbf{x}^*) + R_1(\mathbf{x}^*,\mathbf{h})$.
- **Multivariable, second order (at a stationary point):** $f(\mathbf{x}^*+\mathbf{h})-f(\mathbf{x}^*) = \frac{1}{2}\mathbf{h}^T\nabla^2f(\mathbf{x}^*+\theta\mathbf{h})\,\mathbf{h}$.

---

# Problem-Solving Procedures

### Procedure A — Classify Stationary Points of a Single-Variable Function

1. Compute $f'(x)$.
2. Solve $f'(x)=0$ to find all candidate stationary points.
3. At each candidate $x^*$, compute $f''(x^*)$.
   - If $f''(x^*)>0$: local minimum. Stop.
   - If $f''(x^*)<0$: local maximum. Stop.
   - If $f''(x^*)=0$: continue to the next derivative.
4. Compute successive derivatives $f'''(x^*), f^{(4)}(x^*),\dots$ until you find the first nonzero one, $f^{(n)}(x^*)$.
5. Apply the sufficient-condition theorem (Concept 10):
   - $n$ even, $f^{(n)}(x^*)>0$ → local min.
   - $n$ even, $f^{(n)}(x^*)<0$ → local max.
   - $n$ odd → inflection point (neither).
6. To determine **global** extrema on a closed interval $[a,b]$, also evaluate $f$ at the endpoints $a,b$ (recall: the necessary condition does not apply there) and compare all candidate values.

### Procedure B — Classify Stationary Points of a Multivariable Function

1. Compute $\nabla f(\mathbf{x})$.
2. Solve $\nabla f(\mathbf{x})=0$ (a system of $n$ equations) to find all candidate stationary points.
3. At each candidate $\mathbf{x}^*$, compute the Hessian $H(\mathbf{x}^*) = \nabla^2 f(\mathbf{x}^*)$.
4. Test the definiteness of $H(\mathbf{x}^*)$ (via eigenvalues or leading principal minors):
   - PD → local minimum.
   - ND → local maximum.
   - Indefinite → saddle point.
   - Semi-definite → inconclusive; examine higher-order Taylor terms.

### Procedure C — Solve an Equality-Constrained Problem by Direct Substitution

1. Identify the objective $f(\mathbf{x})$ and the equality constraint(s) $h_i(\mathbf{x})=0$.
2. Solve the constraint(s) explicitly for one variable in terms of the others.
3. Substitute this expression into $f$ to obtain an unconstrained problem in fewer variables.
4. Apply Procedure A (if one variable remains) or Procedure B (if more than one remains).
5. Substitute back to recover all original variables at the optimum.
6. Verify the solution respects the domain restrictions implied by the constraint (e.g., $w>0, h>0$ in Example 2).

---

# Comparisons

| Concept A | Concept B | Key Distinction |
|---|---|---|
| **Local optimum** | **Global optimum** | Local: best among nearby points only. Global: best over the *entire* domain. Every global optimum is local, not conversely. |
| **Necessary condition** ($f'(x^*)=0$ / $\nabla f=0$) | **Sufficient condition** (sign of higher derivative / Hessian definiteness) | Necessary: *must* hold at any extremum, but satisfying it alone doesn't guarantee an extremum exists there ($x^3$ counterexample). Sufficient: if it holds, an extremum is *guaranteed*. |
| **Active constraint** | **Inactive constraint** | Active: $g_j(\mathbf{x})=0$, the constraint boundary is touched. Inactive: $g_j(\mathbf{x})<0$, there is slack. Equality constraints are always active. |
| **Equality constraint** ($h_i(\mathbf{x})=0$) | **Inequality constraint** ($g_j(\mathbf{x})\le 0$) | Equality: no slack allowed, always active. Inequality: slack allowed; may be active or inactive depending on the point. |
| **Positive definite (PD)** | **Positive semi-definite (PSD)** | PD: $\mathbf{h}^TA\mathbf{h}>0$ for *all* $\mathbf{h}\ne0$ (strict). PSD: $\ge 0$ (equality permitted for some $\mathbf{h}\ne 0$) — weaker, and inconclusive for classifying extrema without further tests. |
| **Convex function** | **Convex set** | A convex *function*'s graph lies below its chords ($f(\lambda x+(1-\lambda)y)\le \lambda f(x)+(1-\lambda)f(y)$). A convex *set* contains the entire line segment between any two of its points. Related but distinct: convex optimization requires *both* a convex objective/inequality-constraint functions and a convex feasible set. |
| **Linear Programming** | **Nonlinear Programming** | LP: objective and all constraints are affine. NLP: at least one of the objective or constraint functions is nonlinear (covers convex, quadratic, geometric, and general nonconvex problems). |
| **Classical (calculus-based) methods** | **Modern (heuristic) methods** | Classical: rely on derivatives (gradient, Hessian) and yield provable local (or, for convex problems, global) optimality conditions. Modern: derivative-free, population/nature-inspired search, no formal optimality guarantee but applicable to non-differentiable/black-box/highly non-convex problems. |
| **Saddle point** | **Local extremum (min or max)** | Saddle point: indefinite Hessian, function increases in some directions and decreases in others from the same stationary point. Local extremum: definite Hessian (PD or ND), function moves consistently in one direction (up or down) from the stationary point in *every* direction. |

---

# AI / Machine Learning Connections (Consolidated)

| Optimization Concept | ML / AI Application |
|---|---|
| Unconstrained multivariable optimization, $\nabla f(\mathbf{x}^*)=0$ | Training a neural network: minimizing loss $L(\theta)$ over weights $\theta$ (Example 3a) |
| Quadratic Optimization | SVM optimal-margin hyperplane (Example 3b) |
| Objective/Feasible-region/Local-vs-global | Bias–variance trade-off: minimizing training error vs. generalization error (Example 3c) |
| Regularization (Additional Explanation, Example 3c) | Penalizing model complexity to steer the optimizer toward "Optimal Capacity" rather than zero training loss |
| Positive definite Hessian at a minimum | Confirms a trained model's loss surface is locally "bowl-shaped" around a found solution |
| Convex Optimization | Guarantees that *any* local minimum found (e.g., via gradient descent on a convex loss like ordinary least-squares regression) is the *global* minimum |
| Stochastic Programming (expected-value objectives) | Motivates Stochastic Gradient Descent: minimizing $\mathbb{E}[\text{loss}]$ using random mini-batch estimates |
| Deep Learning Algorithms list (GD, SGD, Momentum, AdaGrad, RMSProp, ADAM) | The family of iterative first-order methods that all target $\nabla f(\mathbf{x}^*)\approx 0$ |
| Modern/heuristic methods (GA, SA, PSO, ACO, ANN, Fuzzy) | Used for non-differentiable or highly non-convex search problems (e.g., neural architecture search, hyperparameter tuning) where classical calculus-based conditions cannot be directly applied |
| Geometric Programming | Device sizing in circuit design (Example 4) — a convex-after-transform engineering optimization problem, structurally related to certain resource-allocation problems in ML systems (e.g., power/latency-constrained model deployment) |

---

# Common Mistakes / Exam Traps

1. **Believing $f'(x^*)=0$ (or $\nabla f(\mathbf{x}^*)=0$) always means an extremum.** *Why it's wrong:* this is only the **necessary** condition; $f(x)=x^3$ at $x=0$ is the standard counterexample (Concept 8, Note 4). *Fix:* always follow up with the sufficient condition (higher derivative test / Hessian definiteness).
2. **Applying the necessary-condition theorem where the derivative doesn't exist.** *Why it's wrong:* the theorem's hypothesis requires $f'(x^*)$ to exist; $f(x)=|x|$ has a minimum at $x=0$ without a defined derivative there (Concept 8, Note 2). *Fix:* check differentiability before invoking the theorem.
3. **Applying the necessary-condition theorem at an interval endpoint.** *Why it's wrong:* only a one-sided derivative can be defined there, so the two-sided-limit proof breaks down (Concept 8, Note 3). *Fix:* separately check endpoint function values when finding a global extremum on a closed interval.
4. **Stopping the higher-order derivative test the first time a derivative is nonzero, without checking whether earlier derivatives were also zero.** *Why it's wrong:* the sufficient-condition theorem (Concept 10) requires *all* derivatives from order $1$ to $n-1$ to be zero and only the $n$-th to be nonzero; skipping this bookkeeping can misidentify $n$'s parity. *Fix:* always differentiate in order, checking each one at $x^*$ before moving to the next.
5. **Confusing positive semi-definite with positive definite.** *Why it's wrong:* PSD only guarantees $\mathbf{h}^TA\mathbf{h}\ge 0$, which is not strong enough to guarantee a strict local minimum (Concept 13); the sufficient-condition theorem requires strict PD. *Fix:* when the Hessian test gives $\succeq 0$ (some eigenvalue exactly $0$), explicitly state the test is inconclusive and further analysis (higher-order Taylor terms) is needed.
6. **Confusing convex functions with convex sets.** *Why it's wrong:* they are related but distinct definitions (a convex function's *graph* behaves in a particular way; a convex set's *points* behave in a particular way) — see the Comparisons table above. *Fix:* always specify whether you're describing a function or a set when using the word "convex."
7. **Assuming touching a constraint boundary automatically makes a point infeasible.** *Why it's wrong:* a "bound acceptable point" (Concept 5's design-space diagram) sits exactly on a constraint boundary yet is fully feasible — the constraint is merely *active*, not violated. *Fix:* feasibility depends on satisfying *every* constraint ($=0$ for equality, $\le 0$ for inequality), not on distance from any one boundary.
8. **Forgetting that equality constraints are always active.** *Why it's wrong:* by definition $h_i(\mathbf{x})=0$ permits no slack, so at every feasible point it is (trivially) satisfied with equality — there is no "inactive" state for an equality constraint (Concept 6). *Fix:* when reasoning about active-constraint sets at a point, automatically include *all* equality constraints, and only check inequality constraints individually for activity.
9. **Treating "local minimum" and "global minimum" as interchangeable.** *Why it's wrong:* outside of convex problems, a function can have many local minima of differing depths (Concept 7's Diagram 1) — a local minimum found by, e.g., gradient descent, need not be the best possible value overall. *Fix:* only claim global optimality when you have additional structure (e.g., convexity) supporting it, or when you've exhaustively compared all candidates (as in a bounded, enumerable problem like Example 1).
10. **Misreading the sign convention when converting a maximization problem.** *Why it's wrong:* forgetting to negate the *optimal value* (not just the objective function) when converting $\max f(\mathbf{x})$ into $-\min[-f(\mathbf{x})]$ form. *Fix:* remember $p^*_{\max} = -p^*_{\min(-f)}$; the sign flips on the final numeric answer too, not just inside the optimization.

---

# Exam Preparation

## What You Must Understand (Conceptual)

- The definition of optimization, and its place within Operations Research (Table 1.1).
- The five phases of an MP/OR study.
- The general constrained optimization problem's components: decision variables, objective, equality/inequality constraints, feasible set.
- Active vs. inactive constraints; why equality constraints are always active.
- Local vs. global minima/maxima, in both one and several variables.
- The distinction between necessary and sufficient optimality conditions.
- What a saddle point is, geometrically and algebraically.
- The defining characteristics of each optimization problem class (LP, unconstrained, convex, quadratic, GP, integer, stochastic, multi-objective, modern/heuristic).

## What You Must Be Able to Derive

- The proof that $f'(x^*)=0$ is necessary for a single-variable local extremum (Concept 8).
- The sign-based completion of the sufficient-condition proof using the parity of $n$ (Concept 10).
- The proof that $\nabla f(\mathbf{x}^*)=0$ is necessary for a multivariable local extremum (Concept 11).
- The proof that a PD Hessian at a stationary point is sufficient for a local minimum (Concept 12).
- The direct-substitution reduction of an equality-constrained problem to an unconstrained one (Example 2/Procedure C).

## What You Must Be Able to Calculate

- Find and classify all stationary points of a given single-variable polynomial (Procedure A; practiced in Concept 9).
- Find and classify all stationary points of a given multivariable function using the gradient and Hessian (Procedure B; practiced in Concept 13's exercise).
- Test whether a given symmetric matrix is PD, ND, PSD, or indefinite, via eigenvalues or leading principal minors.
- Solve a simple equality-constrained optimization problem via direct substitution (Procedure C).
- Enumerate and compare a small finite set of discrete alternatives to find the minimum-cost option (Example 1 style).

## What You Must Be Able to Compare

- Necessary vs. sufficient conditions.
- Local vs. global optimality.
- Active vs. inactive constraints; equality vs. inequality constraints.
- PD vs. PSD vs. ND vs. indefinite Hessians, and their geometric meaning (min/inconclusive/max/saddle).
- LP vs. NLP vs. convex vs. quadratic vs. geometric programming.
- Classical (derivative-based) vs. modern (heuristic) optimization methods.

## Typical Exam Questions

**Conceptual:**
1. State the necessary condition theorem for a local minimum of a single-variable function, including all hypotheses. Why is each hypothesis required?
2. Explain, with an example, why $f'(x^*)=0$ does not guarantee $x^*$ is a local extremum.
3. Define "active constraint" and explain why every equality constraint is active at every feasible point.
4. What structural property distinguishes a Linear Program from a general Nonlinear Program?
5. Explain, in your own words, why convex optimization problems are especially desirable in practice.

**Mathematical:**
6. Prove that if $f(\mathbf{x})$ has a local minimum at $\mathbf{x}^*$ and $\nabla f(\mathbf{x}^*)$ exists, then $\nabla f(\mathbf{x}^*)=0$.
7. State the sufficient-condition theorem for a stationary point of a single-variable function to be a local minimum, maximum, or neither, in terms of the first nonzero derivative $f^{(n)}(x^*)$.
8. Given $A=\begin{pmatrix}4&0\\0&-9\end{pmatrix}$, classify this matrix (PD/ND/PSD/indefinite) and state what this implies about a stationary point with this Hessian.

**Problem-Solving:**
9. Find and classify all stationary points of $f(x) = x^4 - 4x^3 + 6$.
10. Find and classify all stationary points of $f(x_1,x_2) = x_1^2 + x_2^3 - 3x_2$.
11. A farmer has $P$ meters of fencing to enclose a rectangular field bordering a straight river (no fencing needed along the river side). Formulate and solve, via direct substitution, for the dimensions maximizing enclosed area.

**Interpretation:**
12. A contour plot of an objective function shows two separate nested-circle regions and, between them, contour lines that cross in an "X" shape. What do the nested-circle regions represent? What does the "X" crossing represent?
13. In the bias–variance diagram (Example 3c), explain why training error alone is not a reliable guide to choosing model capacity.

## Answer Key

1. See Concept 8 statement: requires $f$ defined on $[a,b]$, local min at interior point $x^*\in(a,b)$, and $f'(x^*)$ exists. The interior requirement is needed because the proof uses a two-sided limit (Concept 8, Note 3); differentiability is needed because the proof manipulates $f'(x^*)$ directly as a well-defined limit (Note 2).
2. See Concept 8, Note 4: $f(x)=x^3$, $f'(0)=0$, but $f$ is increasing through $x=0$ (an inflection point), not a local extremum.
3. See Concept 6: active means $g_j(\mathbf{x})=0$; equality constraints permit no slack by definition, so $h_i(\mathbf{x})=0$ holds exactly at every feasible point, making them trivially always active.
4. See Concept 16.1: in LP, *every* function (objective and all constraints) must be affine; NLP allows at least one nonlinear function.
5. See Concept 16.3 Additional Explanation: convexity guarantees every local minimum is also a global minimum, so the local, calculus-based tools of this chapter become globally reliable.
6. See Concept 11's full step-by-step proof.
7. See Concept 10's theorem statement.
8. Eigenvalues are $4$ and $-9$ (diagonal matrix — eigenvalues are the diagonal entries): mixed sign → **indefinite**. This implies a saddle point (Concept 13).
9. $f'(x) = 4x^3-12x^2 = 4x^2(x-3) = 0 \Rightarrow x=0,3$. $f''(x)=12x^2-24x$. $f''(3)=108-72=36>0\Rightarrow$ local min at $x=3$, $f(3)=81-108+6=-21$. $f''(0)=0\Rightarrow$ check further: $f'''(x)=24x-24$, $f'''(0)=-24\ne0$, order $n=3$ odd $\Rightarrow$ $x=0$ is an inflection point (neither min nor max).
10. $\nabla f = (2x_1,\ 3x_2^2-3)$. Setting to zero: $x_1=0$; $3x_2^2=3\Rightarrow x_2=\pm1$. Candidates: $(0,1)$ and $(0,-1)$. Hessian: $H=\begin{pmatrix}2&0\\0&6x_2\end{pmatrix}$. At $(0,1)$: $H=\begin{pmatrix}2&0\\0&6\end{pmatrix}$, both eigenvalues $>0$ → PD → local min. At $(0,-1)$: $H=\begin{pmatrix}2&0\\0&-6\end{pmatrix}$, mixed signs → indefinite → saddle point.
11. Let $x$ = side perpendicular to the river, $y$ = side parallel to the river. Constraint (fencing used on 2 perpendicular sides + 1 parallel side): $P = 2x+y \Rightarrow y = P-2x$. Objective: maximize $A(x) = x(P-2x) = Px-2x^2$. $A'(x) = P-4x=0 \Rightarrow x=P/4$; $y = P-2(P/4) = P/2$. $A''(x)=-4<0$ → confirmed maximum. (Compare to Example 2's structure — same procedure, different constraint coefficients.)
12. Nested closed-circle contour regions each represent a local extremum (min or max, distinguishable by whether values increase or decrease toward the center). Contour lines crossing in an "X" pattern indicate a saddle point between the two regions (as in the $f(x,y)=x^2-y^2$ example, Concept 13).
13. Because training error decreases (or at worst stays flat) monotonically as capacity increases — a sufficiently flexible model can always fit the training data arbitrarily well, including its noise (overfitting) — so minimizing training error alone pushes capacity toward the overfitting zone rather than toward the Optimal Capacity point that minimizes true generalization error (Example 3c).

---

# Formula Sheet

| # | Formula | Symbols | Applies When | Interpretation |
|---|---|---|---|---|
| 1 | $\min_{\mathbf{x}} f(\mathbf{x})$ s.t. $h_i(\mathbf{x})=0,\ i\in I$; $g_j(\mathbf{x})\le0,\ j\in J$; $\mathbf{x}\in C$ | $f$: objective; $h_i$: equality constraints; $g_j$: inequality constraints; $C$: ambient set | General constrained optimization | The universal template every other problem type in this chapter is a special case of |
| 2 | $F=\{\mathbf{x}\in C\mid h_i(\mathbf{x})=0\ \forall i,\ g_j(\mathbf{x})\le0\ \forall j\}$ | $F$: feasible set | Any constrained problem | The set of all valid candidate solutions |
| 3 | $p^*=\inf\{f(\mathbf{x})\mid \mathbf{x}\in F\}$ | $p^*$: optimal value | Any constrained problem | Best achievable objective value |
| 4 | $g_j(\mathbf{x})=0$ (active) / $g_j(\mathbf{x})<0$ (inactive) | $g_j$: $j$-th inequality constraint | Feasible $\mathbf{x}$ | Whether a constraint boundary is "touched" |
| 5 | $f(x^*)\le f(x^*+h)$, small $h$ | $x^*$: candidate point | Local min test (1 variable) | Definition of local minimum |
| 6 | $f'(x^*)=0$ | $f'$: first derivative | Interior point, $f'$ exists, local extremum | Necessary condition (1 variable) |
| 7 | $f'(x^*)=\dots=f^{(n-1)}(x^*)=0,\ f^{(n)}(x^*)\ne0$ | $n$: order of first nonzero derivative | Classifying a stationary point (1 variable) | $n$ even & $f^{(n)}>0$→min; $n$ even & $f^{(n)}<0$→max; $n$ odd→neither |
| 8 | $f(x^*+h)-f(x^*)=\frac{h^n}{n!}f^{(n)}(x^*+\theta h)$ | $\theta\in(0,1)$ | Proof of formula 7 | Taylor remainder isolates the deciding term |
| 9 | $\nabla f(\mathbf{x}^*)=0$ | $\nabla f$: gradient vector | Interior point, partials exist, local extremum | Necessary condition (multivariable) |
| 10 | $H(\mathbf{x}^*)=\nabla^2f(\mathbf{x}^*)\succ0$ | $H$: Hessian matrix | Stationary point | Sufficient condition for local min |
| 11 | $\mathbf{h}^TA\mathbf{h}>0\ \forall \mathbf{h}\ne0$ | $A$: symmetric matrix | Testing definiteness | Definition of positive definite |
| 12 | $\min_{\mathbf{x}}\mathbf{C}^T\mathbf{x}$ s.t. $A\mathbf{x}=\mathbf{b}$ | $\mathbf{C},A,\mathbf{b}$: fixed data | Linear Programming | All functions affine |
| 13 | $f(\mathbf{x})=a+\mathbf{b}^T\mathbf{x}+\mathbf{x}^TH\mathbf{x}$ | $a,\mathbf{b},H$: fixed data | Quadratic Optimization | Quadratic objective, affine constraints |
| 14 | $f(\mathbf{x})=\sum_{k=1}^K c_kx_1^{r_{1k}}\cdots x_n^{r_{nk}},\ c_k\ge0$ | $c_k,r_{ij}$: fixed data | Geometric Programming | Posynomial functions, $\mathbf{x}>0$ |
| 15 | $x_i\in\mathbb{Z}$ | $\mathbb{Z}$: integers | Integer Programming | Restricts solution to a discrete/finite set |
| 16 | $f(\mathbf{x})=\alpha_1f_1(\mathbf{x})+\alpha_2f_2(\mathbf{x})$ | $\alpha_1,\alpha_2$: weights | Multi-objective Optimization | Weighted-sum scalarization of conflicting objectives |

---

# Concept Map

```text
                        OPTIMIZATION (Mathematical Programming)
                                     |
              -----------------------------------------------------
              |                                                    |
     Operations Research                                General Problem:
     (Table 1.1 taxonomy)                        min f(x) s.t. h_i(x)=0, g_j(x)<=0, x in C
              |                                                    |
     Phases of MP Study                          Feasible Set F --> Active/Inactive constraints
     (Define -> Construct -> Solve ->                              |
      Validate -> Implement)                    Optimal value p*, Optimal set X_opt
                                                                    |
                          -------------------------------------------------------------------
                          |                                                                   |
                 UNCONSTRAINED CASE                                              CONSTRAINED CASE
                          |                                                                   |
        -----------------------------------                          -----------------------------------------
        |                                   |                        |                                        |
  Single Variable                   Multivariable                Equality (h_i=0)                    Inequality (g_j<=0)
        |                                   |                        |                                        |
  Necessary: f'(x*)=0              Necessary: grad f(x*)=0     Direct substitution              Kuhn-Tucker (KKT) conditions
        |                                   |                  Constrained variation                (named, not derived
  Sufficient: f^(n)(x*)<>0,        Sufficient: Hessian PD/ND    Lagrange Multipliers                  in this chapter)
  parity of n --> min/max/neither          |                    (named, not derived
        |                          PD->min, ND->max,             in this chapter)
   Worked Example:                 indefinite->saddle,
   f(x)=12x^5-45x^4+40x^3+5        semi-definite->check
   (min, max, inflection)          higher-order terms
                                            |
                                   Worked Example:
                                   f(x,y)=x^2-y^2 (saddle);
                                   f(x1,x2)=x1^3+x2^3+2x1^2+4x2^2+6
                                            |
                        --------------------------------------------------------------
                        |             |            |             |            |       |
                       LP      Unconstrained     Convex      Quadratic       GP    Integer / Stochastic /
                                                                                     Multi-objective / Modern
                        |
             AI/ML Connections:
             NN training (unconstrained NLP), SVM (Quadratic Opt),
             Regression capacity trade-off (local vs global, regularization),
             Device sizing (Geometric Programming),
             Deep Learning Algorithms (GD/SGD/Momentum/AdaGrad/RMSProp/ADAM
             = iterative search for grad f(x*) = 0)
```

---

# Glossary

- **Active constraint:** An inequality constraint $g_j(\mathbf{x})=0$ satisfied with equality at a feasible point (its boundary is touched).
- **Affine function:** A function of the form $c_0+\mathbf{c}^T\mathbf{x}$ (linear plus constant).
- **Constraint:** A restriction ($h_i(\mathbf{x})=0$ or $g_j(\mathbf{x})\le0$) that a feasible solution must satisfy.
- **Convex function:** A function whose graph lies on or below the chord between any two of its points; formally $f(\lambda x+(1-\lambda)y)\le\lambda f(x)+(1-\lambda)f(y)$.
- **Convex optimization:** A problem where the objective and inequality-constraint functions are convex and the equality constraints are affine.
- **Decision variable:** A quantity $\mathbf{x}$ that the optimizer is free to choose.
- **Embedded optimization:** Optimization run automatically and repeatedly inside a real-time system, with little or no human intervention.
- **Feasible region / feasible set ($F$):** The set of all points satisfying every constraint of the problem.
- **Feasible solution:** Any point $\mathbf{x}\in F$.
- **Geometric Programming (GP):** An optimization problem whose objective and constraints are posynomial functions.
- **Global (absolute) minimum:** A point $x^*$ with $f(x^*)\le f(x)$ for all $x$ in the domain.
- **Gradient ($\nabla f$):** The vector of all first partial derivatives of a multivariable function; points in the direction of steepest ascent.
- **Hessian ($\nabla^2f$):** The matrix of all second partial derivatives of a multivariable function; encodes local curvature.
- **Inactive constraint:** An inequality constraint $g_j(\mathbf{x})<0$ at a feasible point (strict slack remains).
- **Inflection point:** A stationary point that is neither a local min nor a local max (e.g., $x=0$ for $f(x)=x^3$).
- **Infeasible problem/point:** A point (or problem with no points) violating at least one constraint.
- **Integer (discrete) programming:** An optimization problem where some or all decision variables are restricted to integers.
- **Kuhn-Tucker (KKT) conditions:** The named (but, in this chapter, unelaborated) optimality conditions for inequality-constrained optimization problems.
- **Lagrange multipliers:** The named (but, in this chapter, unelaborated) method for solving equality-constrained optimization problems.
- **Level curve / level set / contour:** The set of points $\{\mathbf{x}: f(\mathbf{x})=C\}$ sharing the same objective value $C$.
- **Linear Programming (LP):** An optimization problem in which the objective and all constraints are affine functions.
- **Local (relative) minimum:** A point $x^*$ with $f(x^*)\le f(x^*+h)$ for all sufficiently small $h$.
- **Mathematical Programming (MP):** Another name for optimization; a subfield of Operations Research.
- **Multi-objective optimization:** A problem with more than one objective function to be satisfied simultaneously, often combined via a weighted sum or analyzed via a Pareto front.
- **Necessary condition:** A condition that must hold at any local extremum (e.g., $f'(x^*)=0$), but whose satisfaction alone does not guarantee an extremum.
- **Negative definite (ND) matrix:** A symmetric matrix $A$ with $\mathbf{h}^TA\mathbf{h}<0$ for all $\mathbf{h}\ne0$ (equivalently, all eigenvalues negative).
- **Operations Research (OR):** The broader discipline encompassing mathematical programming, stochastic process techniques, and statistical methods.
- **Optimal point / optimal solution ($\mathbf{x}^*$):** A feasible point achieving the optimal (best) objective value $p^*$.
- **Optimal value ($p^*$):** The best (infimum) achievable value of the objective function over the feasible set.
- **Optimization:** The science of efficiently allocating limited resources to meet objectives subject to constraints; the act of obtaining the best result under given circumstances.
- **Pareto front:** The set of non-dominated solutions in a multi-objective optimization problem, among which improving one objective necessarily worsens another.
- **Parameter:** A fixed, given quantity in a problem's data, as opposed to a decision variable.
- **Posynomial function:** A sum of positive-coefficient monomials $c_k x_1^{r_{1k}}\cdots x_n^{r_{nk}}$ with real (possibly fractional or negative) exponents.
- **Positive definite (PD) matrix:** A symmetric matrix $A$ with $\mathbf{h}^TA\mathbf{h}>0$ for all $\mathbf{h}\ne0$ (equivalently, all eigenvalues positive, or all leading principal minors positive).
- **Positive semi-definite (PSD) matrix:** A symmetric matrix $A$ with $\mathbf{h}^TA\mathbf{h}\ge0$ for all $\mathbf{h}$ (weaker than PD; permits zero eigenvalues).
- **Quadratic optimization:** A problem with a quadratic objective function and affine constraints.
- **Saddle point:** A stationary point at which the Hessian is indefinite (neither PD nor ND); the function increases in some directions and decreases in others.
- **Sensitivity analysis:** Examination of how the optimal solution/value changes as problem data (parameters) change; part of the "Model Solution" phase of an MP study.
- **Stationary point:** A point where the gradient (or, in one variable, the derivative) equals zero; a candidate for a local extremum, but not automatically one.
- **Stochastic programming:** Optimization under uncertainty, where the objective involves an expectation over unknown/random data.
- **Sufficient condition:** A condition that, if satisfied, *guarantees* a stationary point is a particular type of extremum (e.g., a positive definite Hessian guarantees a local minimum).
- **Unconstrained optimization:** An optimization problem with no equality or inequality constraints; only $\min_{\mathbf{x}} f(\mathbf{x})$ over $\mathbf{x}\in\mathbb{R}^n$.

---

# Self-Assessment Questions

1. Write the general constrained optimization problem in full mathematical notation, labeling every symbol.
2. What is the difference between the feasible set $F$ and the optimal set $X_{opt}$?
3. Prove that $f'(x^*)=0$ is necessary for a single-variable local minimum at an interior point where the derivative exists.
4. Give a function for which the necessary condition holds at a point that is *not* a local extremum, and explain why.
5. State the multivariable necessary condition and sketch its proof by contradiction.
6. Given the Hessian $H=\begin{pmatrix}-2&0\\0&-5\end{pmatrix}$ at a stationary point, classify the point.
7. Find and classify the stationary point(s) of $f(x)=x^4$.
8. Explain, using the airfare example, why some optimization problems are solved by enumeration rather than calculus.
9. What change of variables makes a Geometric Program convex? (Conceptual — name the transformation, not the full derivation, since the full derivation is outside this chapter's scope.)
10. Why does the training-error curve in the bias–variance diagram never increase as capacity grows, while the generalization-error curve does?

## Answer Key

1. See Concept 5, "Definition" and "Mathematical Meaning" table.
2. $F$ is the set of *all* feasible points (satisfying the constraints); $X_{opt}\subseteq F$ is the (possibly empty, singleton, or larger) subset of feasible points that additionally achieve the optimal value $p^*$.
3. See Concept 8's full 5-step proof.
4. $f(x)=x^4-4x^3+6x^2-4x$ or, more simply, $f(x)=x^3$ (as in the lecture): $f'(0)=0$ but $x=0$ is an inflection point, not an extremum, because the first nonzero derivative ($f'''(0)$) has odd order.
5. See Concept 11's full 4-step proof (Taylor expansion → dominance of linear term → contradiction via a decreasing direction → conclude $\nabla f(\mathbf{x}^*)=0$).
6. Both diagonal entries (eigenvalues) are negative → **negative definite** → the point is a **local maximum**.
7. $f'(x)=4x^3=0\Rightarrow x=0$. $f''(0)=0$; $f'''(0)=0$; $f^{(4)}(x)=24$, so $f^{(4)}(0)=24\ne0$, order $n=4$ (even), and $f^{(4)}(0)>0$ → **local minimum** at $x=0$ (and, since $f(x)=x^4\ge0=f(0)$ everywhere, it is also the **global** minimum).
8. Because the decision was among a small, finite, non-continuous set of discrete travel plans, not a continuously variable quantity — there was no "derivative" to set to zero; instead, every alternative's cost was simply computed and compared (Concept 2, Example 1).
9. The substitution $x_i = e^{y_i}$ (log-transform of the variables) converts a posynomial (and hence a GP) into a convex function of $\mathbf{y}$ — mentioned in Concept 16.5's Additional Explanation.
10. Training error can always be driven down (or at worst kept flat) by increasing model flexibility, since a more flexible model can fit the specific training points ever more closely, including noise; generalization error, measured on unseen data, initially improves as the model captures true structure but then worsens once the model starts fitting noise specific to the training set rather than the underlying pattern (Example 3c).

---

# Final Review

This chapter established the complete conceptual and mathematical scaffolding for the course:

- **What optimization is** (efficient allocation under constraints, part of Operations Research) and **why it is everywhere** (engineering, finance, embedded systems, and — repeatedly emphasized through the chapter's own examples — machine learning).
- **How to state any optimization problem formally**, using decision variables, an objective function, and equality/inequality constraints, all defined over a feasible set $F$, with well-defined notions of optimal value $p^*$ and optimal point(s) $X_{opt}$.
- **How to test whether a candidate point is actually optimal**, using the necessary condition (zero derivative/gradient) to find *candidates*, and the sufficient condition (sign of a higher derivative, or definiteness of the Hessian) to *classify* each candidate as a minimum, maximum, saddle point, or inflection point — with full proofs for both the single-variable and multivariable cases.
- **How optimization problems are classified** by the mathematical structure of their objective and constraints (LP, convex, quadratic, geometric programming, integer, stochastic, multi-objective, and modern heuristic methods) — a taxonomy that will determine which solution technique (developed in later lectures: simplex, Lagrange multipliers, KKT, etc.) applies to a given problem.

Everything introduced here — the gradient, the Hessian, positive definiteness, active/inactive constraints, and the necessary/sufficient condition framework — is the direct prerequisite machinery for the course's next topics: the method of Lagrange multipliers, the Kuhn-Tucker (KKT) conditions, and the simplex method for Linear Programming, all of which were explicitly *named* in this chapter but deliberately left undeveloped until later lectures.
