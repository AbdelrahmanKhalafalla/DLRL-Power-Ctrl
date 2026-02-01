import random

def fixed_power_policy(state, action_dim):
    """
    Always choose middle power level
    """
    return action_dim // 2


def random_power_policy(state, action_dim):
    """
    Random power selection
    """
    return random.randint(0, action_dim - 1)


def rule_based_policy(state, action_dim, noise=0.1, sinr_threshold=2.0):
    """
    Simple SINR-based heuristic
    """
    channel_gain = state[0]
    interference = state[1]

    assumed_power = 0.3
    sinr_est = (assumed_power * channel_gain) / (interference + noise)

    if sinr_est < sinr_threshold:
        return action_dim - 1  # max power
    else:
        return 0               # min power
