import numpy as np
import itertools
from scipy.spatial import KDTree

def generate_simplex_states(n: int, k: int):
 """
 Generates all integer combinations that sum to n across k bins.
 Used for creating a uniform grid over the (k-1)-dimensional probability simplex.
 """
 for c in itertools.combinations(range(n + k - 1), k - 1):
  yield tuple(b - a - 1 for a, b in zip((-1,) + c, c + (n + k - 1,)))

def precompute_policies(n_states: int, action_discretization: int, n_actions: int):
 """
 Generates the grid of local actions and all possible global policies.
 """
 local_actions = np.array(
  list(generate_simplex_states(action_discretization, n_actions)),
  dtype=float,
 ) / float(action_discretization)

 # A global policy maps each state to a local action (prob distribution over actions)
 global_policies = np.array(
  list(itertools.product(local_actions, repeat=n_states))
 )
 return local_actions, global_policies

def project_to_simplex_grid(mu: np.ndarray, grid: np.ndarray, tree: KDTree) -> int:
 """
 Projects an arbitrary distribution mu onto the closest point in the simplex grid.
 Returns the index of the closest grid point.
 """
 _, idx = tree.query(mu)
 return int(idx)

def compute_W1_1d(mu: np.ndarray, nu: np.ndarray) -> float:
 """
 Computes the 1-Wasserstein distance between two 1D distributions mu and nu.
 """
 cdf_mu = np.cumsum(mu)
 cdf_nu = np.cumsum(nu)
 return np.sum(np.abs(cdf_mu[:-1] - cdf_nu[:-1]))

def compute_wasserstein_power_1d(mu: np.ndarray, nu: np.ndarray, q_norm: int) -> float:
 """
 Computes W_q(mu, nu)^q on the ordered finite state grid with unit spacing.

 The q=1 case is the standard CDF formula. For q>1, W_1(mu, nu)^q is not
 equal to W_q(mu, nu)^q, so we compute the monotone optimal transport cost.
 """
 if q_norm < 1:
  raise ValueError("q_norm must be a positive integer")
 if q_norm == 1:
  return compute_W1_1d(mu, nu)

 i = j = 0
 rem_mu = float(mu[0])
 rem_nu = float(nu[0])
 cost = 0.0
 n = len(mu)

 while i < n and j < n:
  mass = min(rem_mu, rem_nu)
  if mass > 0.0:
   cost += mass * (abs(i - j) ** q_norm)
   rem_mu -= mass
   rem_nu -= mass

  if rem_mu <= 1e-15:
   i += 1
   rem_mu = float(mu[i]) if i < n else 0.0
  if rem_nu <= 1e-15:
   j += 1
   rem_nu = float(nu[j]) if j < n else 0.0

 return cost

def compute_W1_matrix(grid: np.ndarray) -> np.ndarray:
 """
 Computes the pairwise 1-Wasserstein distance matrix for a grid of 1D distributions.
 """
 N = len(grid)
 W1 = np.zeros((N, N))
 for i in range(N):
  for j in range(i + 1, N):
   d = compute_W1_1d(grid[i], grid[j])
   W1[i, j] = d
   W1[j, i] = d
 return W1

def compute_wasserstein_power_matrix(grid: np.ndarray, q_norm: int) -> np.ndarray:
 """
 Computes the pairwise W_q^q cost matrix on the ordered finite state grid.
 """
 if q_norm == 1:
  return compute_W1_matrix(grid)

 N = len(grid)
 Wq_power = np.zeros((N, N))
 for i in range(N):
  for j in range(i + 1, N):
   d = compute_wasserstein_power_1d(grid[i], grid[j], q_norm)
   Wq_power[i, j] = d
   Wq_power[j, i] = d
 return Wq_power

def compute_noise_cost_matrix(noise_grid: np.ndarray, q_norm: int) -> np.ndarray:
 """
 Computes |e - e_tilde|^q on the common-noise grid.

 Scalar noise uses absolute value. Vector-valued noise uses Euclidean norm.
 """
 if q_norm < 1:
  raise ValueError("q_norm must be a positive integer")

 noise = np.asarray(noise_grid, dtype=float)
 if noise.ndim == 1:
  diff = np.abs(noise[:, None] - noise[None, :])
 else:
  diff = np.linalg.norm(noise[:, None, :] - noise[None, :, :], axis=-1)
 return diff ** q_norm
