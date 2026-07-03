"""Pool the 4 GaN 7.5GHz datasets (as-is, no resample) into one split_csv
dataset for OpenDPD. Per-condition 60/20/20 split, then concat train/val/test.
PA-modeling pairs: x = bef_in, y = bef_out (preserve relative amplitude scale).
"""
import os, json, numpy as np

TOP = "/home/mkiuyh/workspace/LLM-basedDPD/datasets"
OUT = "/home/mkiuyh/workspace/LLM-basedDPD/OpenDPD/datasets/GaN_combined"
os.makedirs(OUT, exist_ok=True)

# (config name, data_dir, power_dbm, fs, bw)
CONDS = [
    ("GaN_24dBm",        "GaN_7point5GHz_20MHz",  "24dBm",        128e6, 20e6),
    ("GaN_27dBm",        "GaN_7point5GHz_20MHz",  "27dBm",        128e6, 20e6),
    ("GaN_bw100_20dBm",  "GaN_7point5GHz_100MHz", "20dBm_bw100m", 640e6, 100e6),
    ("GaN_bw100_24dBm",  "GaN_7point5GHz_100MHz", "24dBm_bw100m", 640e6, 100e6),
]
SPLIT = (0.6, 0.2)

def load_iq(dd, pw, prefix):
    d = f"{TOP}/{dd}"
    I = np.loadtxt(f"{d}/{prefix}_{pw}_I.txt"); Q = np.loadtxt(f"{d}/{prefix}_{pw}_Q.txt")
    n = min(len(I), len(Q)); return np.column_stack([I[:n], Q[:n]])

splits = {"train": ([], []), "val": ([], []), "test": ([], [])}
test_src = []
summary = []
for name, dd, pw, fs, bw in CONDS:
    x = load_iq(dd, pw, "bef_in"); y = load_iq(dd, pw, "bef_out")
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    ntr = int(n * SPLIT[0]); nva = int(n * SPLIT[1])
    parts = {"train": (x[:ntr], y[:ntr]),
             "val":   (x[ntr:ntr+nva], y[ntr:ntr+nva]),
             "test":  (x[ntr+nva:], y[ntr+nva:])}
    for s in splits:
        splits[s][0].append(parts[s][0]); splits[s][1].append(parts[s][1])
    test_src.append(np.full(len(parts["test"][0]), name, dtype=object))
    summary.append((name, n, ntr, nva, n-ntr-nva))
    print(f"{name}: N={n}  train={ntr} val={nva} test={n-ntr-nva}  inRMS={np.sqrt(np.mean(x[:,0]**2+x[:,1]**2)):.3f}")

for s in splits:
    X = np.concatenate(splits[s][0]); Y = np.concatenate(splits[s][1])
    np.savetxt(f"{OUT}/{s}_input.csv",  X, delimiter=",", header="I,Q", comments="", fmt="%.10g")
    np.savetxt(f"{OUT}/{s}_output.csv", Y, delimiter=",", header="I,Q", comments="", fmt="%.10g")
    print(f"  {s}: {len(X)} samples written")

# test source labels (per-condition eval)
src = np.concatenate(test_src)
with open(f"{OUT}/test_source.csv", "w") as f:
    f.write("source\n"); [f.write(s + "\n") for s in src]

# spec.json — mixed fs/bw; NMSE is primary, EVM/ACLR evaluated per-condition separately
spec = {
    "description": "Pooled GaN 7.5GHz (24/27dBm 20MHz @128 + 20/24dBm 100MHz @640), as-is (no resample). "
                   "Mixed fs/bw -> NMSE is the primary combined metric; EVM/ACLR are evaluated per condition.",
    "dataset_format": "split_csv",
    "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
    "input_signal_fs": 128e6, "bw_main_ch": 20e6, "n_sub_ch": 10, "nperseg": 2048,
    "pooled_conditions": [c[0] for c in CONDS]
}
with open(f"{OUT}/spec.json", "w") as f:
    json.dump(spec, f, indent=4)
print("\nSpec written. Total train samples:", sum(s[2] for s in summary))
