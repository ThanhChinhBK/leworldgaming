"""PETS — Probabilistic Ensembles with Trajectory Sampling for DareFightingICE.

Discrete-action variant: action embedding + iterated CEM over per-step
``Categorical(num_actions)`` distributions. Reward is computed analytically
from HP primitives — no learned reward head.
"""

from leworldgaming.agents.pets.agent import PETSAgent
from leworldgaming.agents.pets.cem_planner import CEMPlannerDiscrete
from leworldgaming.agents.pets.cost import analytic_reward
from leworldgaming.agents.pets.dynamics import EnsembleDynamics

__all__ = ["PETSAgent", "CEMPlannerDiscrete", "analytic_reward", "EnsembleDynamics"]
