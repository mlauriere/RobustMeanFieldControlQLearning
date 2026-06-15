from abc import ABC, abstractmethod
import numpy as np

class MFCEnvironment(ABC):
 """
 Abstract Base Class for a Mean-Field Control Environment.
 
 Any environment must define:
 - The number of discrete states each agent can be in (`n_states`).
 - The number of discrete actions each agent can take (`n_actions`).
 - The exact mean-field state transition function `transition`.
 - The expected instantaneous mean-field cost function `get_cost`.
 """
 
 @property
 @abstractmethod
 def name(self) -> str:
  """Name of the environment."""
  pass

 @property
 @abstractmethod
 def n_states(self) -> int:
  """Number of discrete states in the single-agent state space."""
  pass

 @property
 @abstractmethod
 def n_actions(self) -> int:
  """Number of discrete actions in the single-agent action space."""
  pass

 @property
 @abstractmethod
 def C_F(self) -> float:
  """Lipschitz constant or robustness penalty multiplier for the environment."""
  pass

 @abstractmethod
 def get_cost(self, mu: np.ndarray, pi: np.ndarray) -> float:
  """
  Computes the instantaneous aggregate cost of the mean-field population.
  
  Args:
   mu: The current population distribution, shape (n_states,).
   pi: The global policy being played, shape (n_states, n_actions),
    where pi[s, a] = P(action=a | state=s).
    
  Returns:
   The scalar cost for this time step.
  """
  pass

 @abstractmethod
 def transition(self, mu: np.ndarray, pi: np.ndarray, e_realized: float) -> np.ndarray:
  """
  Computes the exact deterministic mean-field next state distribution
  given a specific realization of the common noise `e_realized`.
  
  Args:
   mu: The current population distribution, shape (n_states,).
   pi: The global policy being played, shape (n_states, n_actions).
   e_realized: A scalar parameter representing the realized noise 
      for the environment (e.g. beta value).
      
  Returns:
   mu_next: The next population distribution, shape (n_states,).
  """
  pass
