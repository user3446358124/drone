"""Actively stop measured drift when the policy requests holding or turning."""
import math
import numpy as np


def brake_command(command, velocity, heading):
    if velocity is None or heading is None:
        return command
    v = np.asarray(velocity, dtype=float)
    if v.shape != (2,) or not np.all(np.isfinite(v)) or not math.isfinite(heading):
        return command
    if float(np.linalg.norm(v)) < 0.35:
        return command
    c, s = math.cos(heading), math.sin(heading)
    thrust = -np.array([c*v[0]+s*v[1], -s*v[0]+c*v[1]])
    norm = float(np.linalg.norm(thrust))
    if norm > 1.0:
        thrust /= norm
    result = dict(command)
    result['forward'], result['lateral'] = float(thrust[0]), float(thrust[1])
    return result
