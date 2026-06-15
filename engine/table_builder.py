import numpy as np
from scipy.spatial import KDTree
from .mfc_env import MFCEnvironment
from .mfc_utils import (
 generate_simplex_states,
 precompute_policies,
 project_to_simplex_grid,
 compute_W1_matrix,
 compute_noise_cost_matrix,
)

def precompute_tables(env: MFCEnvironment, n_disc: int, a_disc: int, noise_grid: np.ndarray) -> dict:
 """
 Precomputes the transition and reward tensors for a given MFCEnvironment over
 a discrete simplex grid.
 
 Args:
  env: The instantiated MFCEnvironment.
  n_disc: Discretization level for the state simplex.
  a_disc: Discretization level for the action simplex.
  noise_grid: Array of realized noise parameters (e.g. betas).
  
 Returns:
  A dictionary containing the precomputed tables and metadata.
 """
 states = np.array(
  list(generate_simplex_states(n_disc, env.n_states)),
  dtype=float,
 ) / float(n_disc)
 S = len(states)
 grid_tree = KDTree(states)

 if hasattr(env, "precompute_policies"):
  local_actions, global_policies = env.precompute_policies(a_disc)
 else:
  local_actions, global_policies = precompute_policies(env.n_states, a_disc, env.n_actions)
 A = len(global_policies)
 N_E = len(noise_grid)

 print(f"Precomputing {env.name}: S={S}, A={A}, N_E={N_E}")

 transition_table = np.zeros((S, A, N_E), dtype=int)
 reward_table = np.zeros((S, A), dtype=float)

 for s_idx in range(S):
  mu = states[s_idx]
  for a_idx in range(A):
   pi = global_policies[a_idx]
   
   # Convert positive cost to negative reward so the engine maximizes Q
   reward_table[s_idx, a_idx] = -env.get_cost(mu, pi)

   for e_idx in range(N_E):
    e_realized = noise_grid[e_idx]
    
    # Get true mean-field next state
    mu_next = env.transition(mu, pi, e_realized)
    
    # Project back to grid
    transition_table[s_idx, a_idx, e_idx] = project_to_simplex_grid(mu_next, states, grid_tree)

 W1 = compute_W1_matrix(states)
 noise_grid = np.asarray(noise_grid, dtype=float)

 return {
  "states": states,
  "S": S,
  "local_actions": local_actions,
  "global_policies": global_policies,
  "A": A,
  "transition_table": transition_table,
  "reward_table": reward_table,
  "W1": W1,
  "noise_grid": noise_grid,
  "noise_cost_power": {1: compute_noise_cost_matrix(noise_grid, 1)},
  "grid_tree": grid_tree
 }
