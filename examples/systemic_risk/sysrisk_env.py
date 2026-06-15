import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from engine.mfc_env import MFCEnvironment

class SysRiskModularEnv(MFCEnvironment):
 """
 Systemic risk model for a stylized interbank lending system.
 
 State space: 0 = distressed (bailout cost), 1 = normal, 2 = well-capitalized
 Action space: 0 = drift down (-1), 1 = do nothing (0), 2 = drift up (+1)
 Common noise: e = -1, 0, +1 (market shock)
 """

 def __init__(self, cliff_penalty: float = 2.0):
  self._C_F = 1.0
  self.cliff_penalty = cliff_penalty
  self.mu_target = np.array([0.0, 1.0, 0.0])
  self.action_values = [-1, 0, 1]

 @property
 def name(self) -> str:
  return "Systemic_Risk_Modular"

 @property
 def n_states(self) -> int:
  return 3

 @property
 def n_actions(self) -> int:
  return 3

 @property
 def C_F(self) -> float:
  return self._C_F

 def get_cost(self, mu: np.ndarray, pi: np.ndarray) -> float:
  """
  Cost: ||mu - mu_target||^2 + cliff_penalty * mu[0]
  """
  return float(np.sum((mu - self.mu_target) ** 2) + self.cliff_penalty * mu[0])

 def transition(self, mu: np.ndarray, pi: np.ndarray, e_realized: float) -> np.ndarray:
  """
  Computes the exact deterministic mean-field next state.
  e_realized is the market shock: -1, 0, or +1.
  """
  mu_next = np.zeros(self.n_states)
  for x in range(self.n_states):
   for act_idx, a_val in enumerate(self.action_values):
    prob_a = pi[x, act_idx]
    if prob_a > 0:
     # Next state is index x (0-indexed) shifted by action and noise
     # The state physical value is x + 1 (1, 2, 3)
     s_next = (x + 1) + a_val + e_realized
     s_next = max(1, min(self.n_states, s_next))
     t_idx = int(s_next - 1)
     mu_next[t_idx] += mu[x] * prob_a
  return mu_next
