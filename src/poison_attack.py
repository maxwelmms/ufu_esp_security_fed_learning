import numpy as np

def label_poisoning(X, y, poison_fraction=0.1):
    """
    Aplica ataque de label poisoning trocando aleatoriamente labels de uma fração dos dados.
    """
    y_poisoned = y.copy()
    n_samples = int(len(y) * poison_fraction)
    indices = np.random.choice(len(y), n_samples, replace=False)
    y_poisoned.iloc[indices] = 1 - y_poisoned.iloc[indices]  # inverter label binária (0 -> 1, 1 -> 0)
    return X, y_poisoned
