import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from engine.mfc_env import MFCEnvironment
from engine.mfc_utils import generate_simplex_states

class SEIRModularEnv(MFCEnvironment):
 """
 SEIR Epidemic Environment for Mean-Field Control.
 State space: 0 = Susceptible (S), 1 = Exposed (E), 2 = Infected (I), 3 = Recovered (R)
 Action space: 0 = Unrestricted (U), 1 = Distancing (D)
 """

 def __init__(
  self,
  sigma=0.5,
  gamma=0.2,
  cost_distancing=0.5,
  distancing_efficacy=1.0,
  lipschitz_constant=3.2,
 ):
  self._C_F = lipschitz_constant
  self.sigma = sigma
  self.gamma = gamma
  self.cost_distancing = cost_distancing
  self.distancing_efficacy = distancing_efficacy

 @property
 def name(self) -> str:
  return "SEIR_Epidemic_Modular"

 @property
 def n_states(self) -> int:
  return 4

 @property
 def n_actions(self) -> int:
  return 2

 @property
 def C_F(self) -> float:
  return self._C_F

 def precompute_policies(self, action_discretization: int):
  """
  Reduced policy class: only susceptible agents choose a distancing level.
  In this SEIR dynamics, the action enters only through the susceptible
  transition S->E; distancing E/I/R agents only adds cost.
  """
  local_actions = np.array(
   list(generate_simplex_states(action_discretization, self.n_actions)),
   dtype=float,
  ) / float(action_discretization)

  global_policies = []
  no_distancing = np.array([1.0, 0.0])
  for susceptible_action in local_actions:
   global_policies.append(
    np.array([susceptible_action, no_distancing, no_distancing, no_distancing])
   )
  return local_actions, np.array(global_policies)

 def get_cost(self, mu: np.ndarray, pi: np.ndarray) -> float:
  """
  Cost = P(I) + cost_distancing * P(D)
  """
  p_I = mu[2]
  p_D = np.sum(mu * pi[:, 1])
  return p_I + self.cost_distancing * p_D

 def transition(self, mu: np.ndarray, pi: np.ndarray, e_realized: float) -> np.ndarray:
  """
  Computes the exact deterministic mean-field next state.
  e_realized is the transmission rate beta.
  """
  effective_susceptible = pi[0, 0] + (1.0 - self.distancing_efficacy) * pi[0, 1]
  trans_S_E = min(mu[0], e_realized * mu[2] * mu[0] * effective_susceptible)
  trans_E_I = self.sigma * mu[1]
  trans_I_R = self.gamma * mu[2]
  
  mu_next = np.array([
   mu[0] - trans_S_E,
   mu[1] + trans_S_E - trans_E_I,
   mu[2] + trans_E_I - trans_I_R,
   mu[3] + trans_I_R
  ])
  mu_next = np.clip(mu_next, 0.0, 1.0)
  
  # Ensure it sums to exactly 1
  total = np.sum(mu_next)
  if total > 0:
   mu_next = mu_next / total
  
  return mu_next
