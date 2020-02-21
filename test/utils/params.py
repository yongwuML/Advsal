import random


class TrackerParams:
    """Class for tracker parameters."""
    def set_default_values(self, default_vals: dict):
        for name, val in default_vals.items():
            if not hasattr(self, name):
                setattr(self, name, val)


def Choice(*args):
    """Can be used to sample random parameter values."""
    return random.choice(args)
