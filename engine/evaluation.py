import numpy as np

def evaluate_policy_exact(
 tables: dict,
 Q: np.ndarray,
 p_true: np.ndarray,
 discount: float,
 tol: float = 1e-10,
 max_iter: int = 10000,
 init_dist: np.ndarray = None,
) -> float:
 """
 Deterministically evaluates the infinite-horizon discounted cost of the
 greedy policy induced by Q on the finite projected state grid.

 By default, this matches the uniform random initial state used in the numerical experiments,
 but removes Monte Carlo noise and finite-horizon truncation bias.
 """
 S = tables["S"]
 p_true = np.asarray(p_true, dtype=float)
 p_true = p_true / np.sum(p_true)

 best_actions = np.argmax(Q, axis=1)
 state_to_next = tables["transition_table"][np.arange(S), best_actions, :]
 state_to_cost = -tables["reward_table"][np.arange(S), best_actions]

 V = np.zeros(S, dtype=float)
 for _ in range(max_iter):
  expected_next = np.zeros(S, dtype=float)
  for e_idx, prob in enumerate(p_true):
   expected_next += prob * V[state_to_next[:, e_idx]]

  V_new = state_to_cost + discount * expected_next
  if np.max(np.abs(V_new - V)) < tol:
   V = V_new
   break
  V = V_new

 if init_dist is None:
  return float(np.mean(V))

 init_dist = np.asarray(init_dist, dtype=float)
 init_dist = init_dist / np.sum(init_dist)
 return float(np.dot(init_dist, V))


def evaluate_policy_vectorized(
 tables: dict,
 Q: np.ndarray,
 p_true: np.ndarray,
 discount: float,
 n_sims: int = 5000,
 T_max: int = 50,
 seed: int = 42,
) -> float:
 """
 Evaluates the discounted infinite horizon cost of a policy (derived from Q)
 by running multiple independent simulations in parallel.
 
 Args:
  tables: Precomputed dict with transition_table, reward_table, etc.
  Q: The optimal Q-value table, shape (S, A).
  p_true: The true underlying noise distribution.
  discount: The discount factor gamma.
  n_sims: Number of independent simulation trajectories to run.
  T_max: Truncation horizon for evaluation.
  seed: Random seed for reproducibility.
  
 Returns:
  The expected discounted cost.
 """
 np.random.seed(seed)
 S, A = tables["S"], tables["A"]
 N_E = len(p_true)
 
 # Deriving deterministic policy from Q (maximize reward)
 best_actions = np.argmax(Q, axis=1)
 
 # Create fast lookup tables specifically for the chosen actions
 state_to_next = tables["transition_table"][np.arange(S), best_actions, :]
 
 # We want to return the POSITIVE cost. tables["reward_table"] stores NEGATIVE cost.
 state_to_cost = -tables["reward_table"][np.arange(S), best_actions]
 
 # Initialize all simulations from random starting states
 current_states = np.random.randint(0, S, size=n_sims)
 total_cost = 0.0
 gamma = 1.0
 
 for t in range(T_max):
  total_cost += gamma * np.mean(state_to_cost[current_states])
  gamma *= discount
  
  # Sample realization for each trajectory
  noise = np.random.choice(N_E, size=n_sims, p=p_true)
  current_states = state_to_next[current_states, noise]
  
 return total_cost
