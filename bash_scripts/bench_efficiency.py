import os,sys,time,numpy as np,torch
ROOT="/home/mkiuyh/workspace/LLM-basedDPD/OpenDPD"; sys.path.insert(0,ROOT); os.chdir(ROOT)
import models as M
torch.backends.cudnn.benchmark=True
dev=torch.device('cuda:0')

def build(arch):
    if arch=='DGRU': return M.CoreModel(input_size=2,hidden_size=8,num_layers=1,backbone_type='dgru')
    if arch=='Mamba': return M.CoreModel(input_size=2,hidden_size=6,num_layers=1,backbone_type='mamba',mamba_d_state=4,mamba_d_conv=4,mamba_expand=2)
    if arch=='Transformer': return M.CoreModel(input_size=2,hidden_size=6,num_layers=1,backbone_type='transformer',n_heads=2,d_ff=18,use_pos_encoding=0)

def bench(model,FL,B,dev,iters=50,warm=10):
    model.eval()
    x=torch.randn(B,FL,2,device=dev)
    with torch.no_grad():
        for _ in range(warm): model(x)
        if dev.type=='cuda': torch.cuda.synchronize()
        t=[]
        for _ in range(iters):
            s=time.perf_counter(); model(x)
            if dev.type=='cuda': torch.cuda.synchronize()
            t.append(time.perf_counter()-s)
    t=np.median(t); return t*1000, B/t   # ms/batch, windows/sec

ARCHS=['DGRU','Mamba','Transformer']
params={a:sum(p.numel() for p in build(a).parameters()) for a in ARCHS}
print("params:",params)
print(f"\n=== GPU 지연(ms/batch, B=256) & 처리량(k win/s) ===")
print(f"{'FL':>6}"+"".join(f"{a:>26}" for a in ARCHS))
for FL in [50,200,1000]:
    row=f"{FL:>6}"
    for a in ARCHS:
        m=build(a).to(dev)
        ms,wps=bench(m,FL,256,dev)
        row+=f"   {ms:7.2f}ms /{wps/1e3:7.1f}k/s"
    print(row)
print(f"\n=== 단일 윈도우(B=1) 지연 [실시간 스트리밍 관점] ===")
print(f"{'FL':>6}"+"".join(f"{a:>16}" for a in ARCHS))
for FL in [50,200,1000]:
    row=f"{FL:>6}"
    for a in ARCHS:
        m=build(a).to(dev); ms,_=bench(m,FL,1,dev,iters=100)
        row+=f"   {ms:9.3f}ms"
    print(row)
