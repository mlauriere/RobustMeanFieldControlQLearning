# Algorithm Mapping

The public code implements the asynchronous version of Algorithm 1 from the
paper. Below is a conceptual mapping.

## Key Components

### Finite Projected State Grid `\check S`

Uniform discretization of the probability simplex over the agent state space.
In `engine/table_builder.py`, `generate_simplex_states(n_disc, n_states)`
produces all integer compositions of `n_disc` into `n_states` bins, normalized
to sum to 1.

### Finite Policy Grid `\check \Pi`

Discretized local actions (probability distributions over agent-level actions)
combined into global policies. `precompute_policies()` in
`engine/mfc_utils.py` builds the action grid. For SIS and SEIR, the
policy space is reduced to exclude actions that only increase cost without
affecting dynamics (see `precompute_policies` in each environment).

### Common-Noise Robust Dual

The robust Bellman target uses a common-noise Wasserstein penalty. For
robustness radius `m` and Wasserstein exponent `q`, the penalty is `m^q * lambda`.
The dual is computed by maximizing over the lambda grid:

```
phi(s, a) = max_{lambda >= 0} [ min_{e_tilde} J(s', e_tilde)
             - m^q * lambda ]
```

where the inner minimization includes the transport cost
`lambda * |e - e_tilde|^q`.

Implementation: `compute_robust_bellman_target()` in
`engine/robust_solvers.py`.

### Asynchronous Q-Learning (Algorithm 1)

- A permutation of all state-action pairs is randomly shuffled.
- For each pair `(s, a)`, one common-noise realization `e` is drawn from `\hat p`.
- The robust Bellman target is computed using that single sample.
- Q is updated with a decreasing learning rate `(visits + 1)^{-w}`.
- After exhausting the permutation, a new random permutation is drawn.
- This repeats for the total update budget.

Implementation: `run_async_qlearning()` in `engine/robust_solvers.py` and
`run_sampled_qlearning_trace()` in `engine/convergence.py`.

### Idealized Bellman Iteration

As a model-known reference, deterministic Bellman iteration is run on the
same finite grid using the full reference law `\hat p` (no sampling). This
yields the finite-grid fixed point `Q_m^*`.

Implementation: `run_value_iteration_trace()` in `engine/convergence.py`.

## Notation Map

| Paper | Code |
|---|---|
| `\check S` | Projected lifted-state grid (`tables["states"]`) |
| `S` | Number of lifted states (`tables["S"]`) |
| `\check \Pi` | Finite policy grid (`tables["global_policies"]`) |
| `A` | Number of policies (`tables["A"]`) |
| `m` | Robustness radius (`robust_m`) |
| `q` | Wasserstein exponent (`q_norm`) |
| `\mathbf{\hat p}_{arepsilon^0}` | Reference common-noise law (`p_hat`) |
| `\check Q_T` | Learned finite-grid Q-table (`Q`) |
| `\check Q_m^*` | Idealized finite-grid fixed point (`Q_ref`) |
| `\gamma` | Discount factor (`discount`) |
| `\lambda`-grid | Dual variable grid (`lambda_grid`) |
| `w` | Learning-rate exponent (`w_lr`) |
