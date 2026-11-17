import numpy as np
import pandas as pd
from src.game_theory import oracle_benchmark as ob

ob.MODEL = ob.load_model()
EDGES = np.arange(850, 921, 10)

def run(n=300, seed=1):
    rng = np.random.default_rng(seed)
    bin_idx = np.digitize(ob.GRID, EDGES) - 1
    rows = []
    for _ in range(n):
        case = ob.make_case(rng)
        tp, mp = ob.true_p(case), ob.model_p(case)
        for k in range(len(EDGES) - 1):
            m = bin_idx == k
            rows.append(dict(bin=f"{EDGES[k]}-{EDGES[k+1]}",
                             arch=case["s"]["archetype"],
                             true=tp[m].mean(), model=mp[m].mean()))
    df = pd.DataFrame(rows)
    for key in ["bin", "arch"]:
        t = df.groupby(key)[["true", "model"]].mean()
        t["bias"] = t["model"] - t["true"]
        print(t.round(3), "\n")

if __name__ == "__main__":
    run()