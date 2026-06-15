import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from engine.mfc_env import MFCEnvironment
from engine.mfc_utils import generate_simplex_states

class SISModularEnv(MFCEnvironment):
 """
 SIS Epidemic Environment for Mean-Field Control.
 State space: 0 = Susceptible (S), 1 = Infected (I)
 Action space: 0 = Unrestricted (U), 1 = Distancing (D)
 """

 def __init__(self, cost_distancing=0.5, lipschitz_constant=1.8):
  self._C_F = lipschitz_constant
  self.cost_distancing = cost_distancing

 @property
 def name(self) -> str:
  return "SIS_Epidemic_Modular"

 @property
 def n_states(self) -> int:
  return 2

 @property
 def n_actions(self) -> int:
  return 2

 @property
 def C_F(self) -> float:
  return self._C_F

 def precompute_policies(self, action_discretization: int):
  """
  Reduced policy class: only susceptible agents choose a distancing level.
  Distancing infected agents has no effect in this SIS dynamics and only
  adds cost, so it is removed from the tabular action space.
  """
  local_actions = np.array(
   list(generate_simplex_states(action_discretization, self.n_actions)),
   dtype=float,
  ) / float(action_discretization)

  global_policies = []
  infected_no_distancing = np.array([1.0, 0.0])
  for susceptible_action in local_actions:
   global_policies.append(
    np.array([susceptible_action, infected_no_distancing])
   )
  return local_actions, np.array(global_policies)

 def get_cost(self, mu: np.ndarray, pi: np.ndarray) -> float:
  """
  Cost = P(I) + cost_distancing * P(D)
  """
  p_I = mu[1]
  p_D = mu[0]*pi[0, 1] + mu[1]*pi[1, 1]
  return p_I + self.cost_distancing * p_D

 def transition(self, mu: np.ndarray, pi: np.ndarray, e_realized: float) -> np.ndarray:
  """
  Computes the exact deterministic mean-field next state.
  e_realized is the transmission rate beta.
  """
  new_I = 0.7 * mu[1] + e_realized * mu[1] * mu[0] * pi[0, 0]
  new_I = np.clip(new_I, 0.0, 1.0)
  
  mu_next = np.array([1.0 - new_I, new_I])
  return mu_next
