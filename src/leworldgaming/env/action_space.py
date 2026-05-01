"""Discrete action space for DareFightingICE.

Each integer encodes a button combo. The exact set comes from pyftg's Action
enum — wire it up during weekend dev. We pin a placeholder size here so model
shapes are stable across the codebase.
"""

NUM_ACTIONS = 56  # Placeholder; actual count comes from pyftg.struct.action.Action
