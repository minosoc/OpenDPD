"""Leave-one-condition-out datasets for zero-shot generalization study.
For each held-out condition D: train/val = pooled other 3 conditions, test = D.
Test-set is a single condition -> its fs/bw are well-defined (clean zero-shot metrics)."""
import os, json, numpy as np
TOP="/home/mkiuyh/workspace/LLM-basedDPD/datasets"
OUTROOT="/home/mkiuyh/workspace/LLM-basedDPD/OpenDPD/datasets"
# name, data_dir, power, fs, bw
C=[("c24","GaN_7point5GHz_20MHz","24dBm",128e6,20e6),
   ("c27","GaN_7point5GHz_20MHz","27dBm",128e6,20e6),
   ("b20","GaN_7point5GHz_100MHz","20dBm_bw100m",640e6,100e6),
   ("b24","GaN_7point5GHz_100MHz","24dBm_bw100m",640e6,100e6)]
def load(dd,pw,pfx):
    I=np.loadtxt(f"{TOP}/{dd}/{pfx}_{pw}_I.txt");Q=np.loadtxt(f"{TOP}/{dd}/{pfx}_{pw}_Q.txt")
    n=min(len(I),len(Q));return np.column_stack([I[:n],Q[:n]]).astype(np.float32)
# cache per-condition splits
data={}
for name,dd,pw,fs,bw in C:
    x=load(dd,pw,"bef_in"); y=load(dd,pw,"bef_out"); n=min(len(x),len(y)); x,y=x[:n],y[:n]
    ntr=int(n*0.6); nva=int(n*0.2)
    data[name]=dict(xtr=x[:ntr],ytr=y[:ntr],xva=x[ntr:ntr+nva],yva=y[ntr:ntr+nva],
                    xte=x[ntr+nva:],yte=y[ntr+nva:],fs=fs,bw=bw)
for held,dd,pw,fs,bw in C:
    others=[c[0] for c in C if c[0]!=held]
    out=f"{OUTROOT}/GaN_loo_{held}"; os.makedirs(out,exist_ok=True)
    Xtr=np.concatenate([data[o]['xtr'] for o in others]); Ytr=np.concatenate([data[o]['ytr'] for o in others])
    Xva=np.concatenate([data[o]['xva'] for o in others]); Yva=np.concatenate([data[o]['yva'] for o in others])
    Xte,Yte=data[held]['xte'],data[held]['yte']   # held-out condition test = zero-shot
    for nm,(A,B) in [('train',(Xtr,Ytr)),('val',(Xva,Yva)),('test',(Xte,Yte))]:
        np.savetxt(f"{out}/{nm}_input.csv",A,delimiter=",",header="I,Q",comments="",fmt="%.10g")
        np.savetxt(f"{out}/{nm}_output.csv",B,delimiter=",",header="I,Q",comments="",fmt="%.10g")
    spec=dict(description=f"Leave-one-out: train=pool({'+'.join(others)}), test=zero-shot {held}",
              dataset_format="split_csv",split_ratios=dict(train=0.6,val=0.2,test=0.2),
              input_signal_fs=fs,bw_main_ch=bw,n_sub_ch=10,nperseg=2048,held_out=held)
    json.dump(spec,open(f"{out}/spec.json","w"),indent=4)
    print(f"GaN_loo_{held}: train={len(Xtr)} val={len(Xva)} test(held {held})={len(Xte)} | test fs={fs/1e6:.0f}MHz bw={bw/1e6:.0f}MHz")
