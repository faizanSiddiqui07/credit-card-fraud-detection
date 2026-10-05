import numpy as np

V_COLS = [f"V{i}" for i in range(1, 29)]


def make_features(Xd):
    """Same feature recipe used during training. Needs Time, V1..V28, Amount."""
    o = Xd[V_COLS].copy()
    o["logAmt"] = np.log1p(Xd["Amount"])
    hour = (Xd["Time"] / 3600.0) % 24
    o["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    o["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    o["Vnorm"] = np.sqrt((Xd[V_COLS] ** 2).sum(axis=1))
    o["Vabsmax"] = Xd[V_COLS].abs().max(axis=1)
    return o
