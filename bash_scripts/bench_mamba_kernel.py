import os,sys,time,numpy as np,torch
ROOT="/home/mkiuyh/workspace/LLM-basedDPD/OpenDPD"; sys.path.insert(0,ROOT); os.chdir(ROOT)
import models as M
from backbones.mamba import MambaBlock
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
torch.backends.cudnn.benchmark=True
dev=torch.device('cuda:0')

def kernel_scan(self, x, delta, A, B, C, D):
    # ours: x/delta (B,L,Din), B/C (B,L,N) -> kernel: (B,Din,L)/(B,N,L); softplus already applied upstream
    y = selective_scan_fn(x.transpose(1,2).contiguous(), delta.transpose(1,2).contiguous(),
                          A, B.transpose(1,2).contiguous(), C.transpose(1,2).contiguous(),
                          D.float(), z=None, delta_bias=None, delta_softplus=False)
    return y.transpose(1,2)

def build(arch):
    if arch=='DGRU': return M.CoreModel(input_size=2,hidden_size=8,num_layers=1,backbone_type='dgru')
    return M.CoreModel(input_size=2,hidden_size=6,num_layers=1,backbone_type='mamba',
                       mamba_d_state=4,mamba_d_conv=4,mamba_expand=2)

# ---- 1) numerical equivalence ----
torch.manual_seed(0)
m=build('Mamba').to(dev).eval()
x=torch.randn(4,128,2,device=dev)
with torch.no_grad():
    ref=m(x).clone()
    orig=MambaBlock._selective_scan
    MambaBlock._selective_scan=kernel_scan
    ker=m(x).clone()
    MambaBlock._selective_scan=orig
diff=(ref-ker).abs().max().item()
print(f"[equivalence] max|Δ|={diff:.3e} -> {'OK' if diff<1e-3 else 'MISMATCH!'}")

# ---- 2) benchmark ----
def bench(model,FL,B,iters=50,warm=10):
    x=torch.randn(B,FL,2,device=dev)
    with torch.no_grad():
        for _ in range(warm): model(x)
        torch.cuda.synchronize()
        ts=[]
        for _ in range(iters):
            s=time.perf_counter(); model(x); torch.cuda.synchronize(); ts.append(time.perf_counter()-s)
    t=np.median(ts); return t*1000, B/t

rows={}
for name in ['Mamba(PyTorch)','Mamba(kernel)','DGRU']:
    if name=='Mamba(kernel)': MambaBlock._selective_scan=kernel_scan
    else: MambaBlock._selective_scan=orig
    mod=build('DGRU' if name=='DGRU' else 'Mamba').to(dev).eval()
    for FL in [50,200,1000]:
        ms,wps=bench(mod,FL,256)
        ms1,_=bench(mod,FL,1,iters=100)
        rows[(name,FL)]=(ms,wps,ms1)
MambaBlock._selective_scan=orig

print(f"\n{'model':>16} {'FL':>6} {'B=256 ms':>10} {'k win/s':>9} {'B=1 ms':>8}")
for name in ['Mamba(PyTorch)','Mamba(kernel)','DGRU']:
    for FL in [50,200,1000]:
        ms,wps,ms1=rows[(name,FL)]
        print(f"{name:>16} {FL:>6} {ms:>10.2f} {wps/1e3:>9.1f} {ms1:>8.3f}")
print("\nspeedup (kernel vs PyTorch), throughput:")
for FL in [50,200,1000]:
    print(f"  FL={FL:5d}: x{rows[('Mamba(kernel)',FL)][1]/rows[('Mamba(PyTorch)',FL)][1]:.2f}")
