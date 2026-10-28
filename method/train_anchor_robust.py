#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
N_CLASSES = 10
N_VIEWS = 9
CELL_AREA = 80.0 * 80.0
PATCH_AREA = 320.0 * 320.0

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def feature_columns(df):
    return [
        c for c in df.columns
        if len(c) >= 4 and c[0] == "c" and c[1:3].isdigit() and c[3] == "_"
    ]

def invariant_from_z0(g):
    count_cols = [f"c{i:02d}_count_density" for i in range(16)]
    cover_cols = [f"c{i:02d}_footprint_density" for i in range(16)]
    comp_cols = [f"c{i:02d}_mean_compactness" for i in range(16)]
    counts = g[count_cols].to_numpy(np.float32) * CELL_AREA / 10000.0
    footprint = g[cover_cols].to_numpy(np.float32) * CELL_AREA
    mean_comp = g[comp_cols].to_numpy(np.float32)
    n = counts.sum(axis=1)
    fp = footprint.sum(axis=1)
    weighted_comp = (mean_comp * counts).sum(axis=1)
    return np.column_stack([
        n / PATCH_AREA * 10000.0,
        fp / PATCH_AREA,
        np.divide(fp, n, out=np.zeros_like(fp), where=n > 0),
        np.divide(weighted_comp, n, out=np.zeros_like(weighted_comp), where=n > 0),
    ]).astype(np.float32)

def grouped_arrays(df, cities):
    d = df[df.city.isin(cities)].copy()
    feat = feature_columns(d)
    zorder = [f"z{i}" for i in range(N_VIEWS)]

    z0 = d[d.zoning == "z0"].sort_values(["city","split","sample_index"]).reset_index(drop=True)
    keys = z0[["city","split","sample_index"]].copy()
    y = z0.label.to_numpy(np.int64) - 1
    global_x = invariant_from_z0(z0)

    views = []
    for z in zorder:
        gz = d[d.zoning == z].sort_values(["city","split","sample_index"]).reset_index(drop=True)
        if len(gz) != len(z0):
            raise ValueError(f"View {z} length mismatch")
        if not gz[["city","split","sample_index"]].equals(keys):
            raise ValueError(f"View {z} keys mismatch")
        views.append(gz[feat].to_numpy(np.float32))
    local_x = np.stack(views, axis=1)
    return keys, global_x, local_x, y

class DualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_net = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, N_CLASSES),
        )
        self.local_net = nn.Sequential(
            nn.Linear(64, 96), nn.ReLU(), nn.LayerNorm(96),
            nn.Dropout(0.10),
            nn.Linear(96, 64), nn.ReLU(),
            nn.Linear(64, N_CLASSES),
        )
        self.residual_log_scale = nn.Parameter(torch.tensor(-0.7))

    def forward(self, gx, lx):
        gl = self.global_net(gx)
        b, k, f = lx.shape
        ll = self.local_net(lx.reshape(b*k, f)).reshape(b, k, N_CLASSES)
        scale = torch.sigmoid(self.residual_log_scale)
        logits = gl[:,None,:] + scale * ll
        return gl, logits

def standardize(train_g, train_l, test_g, test_l):
    gm = train_g.mean(0, keepdims=True); gs = train_g.std(0, keepdims=True) + 1e-6
    lm = train_l.reshape(-1, train_l.shape[-1]).mean(0, keepdims=True)
    ls = train_l.reshape(-1, train_l.shape[-1]).std(0, keepdims=True) + 1e-6
    return (
        (train_g-gm)/gs,
        (train_l-lm[None,:,:])/ls[None,:,:],
        (test_g-gm)/gs,
        (test_l-lm[None,:,:])/ls[None,:,:],
    )

def class_weights(y):
    counts=np.bincount(y,minlength=N_CLASSES).astype(np.float32)
    w=np.zeros(N_CLASSES,dtype=np.float32)
    present=counts>0
    w[present]=len(y)/(N_CLASSES*counts[present])
    if present.any():
        w[present] = np.minimum(w[present], np.quantile(w[present],0.9)*1.5)
    return torch.tensor(w,dtype=torch.float32)

def entropy_confidence(logits):
    p=torch.softmax(logits,dim=-1)
    ent=-(p*torch.log(p+1e-8)).sum(-1)
    return (1.0-ent/math.log(N_CLASSES)).clamp(0,1)

def symmetric_kl(p_logits, q_logits):
    p=torch.softmax(p_logits,dim=-1)
    q=torch.softmax(q_logits,dim=-1)
    lp=torch.log_softmax(p_logits,dim=-1)
    lq=torch.log_softmax(q_logits,dim=-1)
    return 0.5*((p*(lp-lq)).sum(-1)+(q*(lq-lp)).sum(-1))

def train_one(train_g, train_l, train_y, strategy, device):
    set_seed()
    model=DualModel().to(device)
    w=class_weights(train_y).to(device)
    ce=nn.CrossEntropyLoss(weight=w,reduction="none")
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)

    ds=TensorDataset(
        torch.tensor(train_g,dtype=torch.float32),
        torch.tensor(train_l,dtype=torch.float32),
        torch.tensor(train_y,dtype=torch.long),
    )
    loader=DataLoader(ds,batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(SEED))

    best=None
    best_loss=float("inf")
    patience=15
    bad=0

    for epoch in range(120):
        model.train()
        total=0.0
        n=0
        for gx,lx,y in loader:
            gx,lx,y=gx.to(device),lx.to(device),y.to(device)
            gl,logits=model(gx,lx)
            b,k,c=logits.shape

            global_loss=ce(gl,y).mean()
            view_losses=ce(logits.reshape(b*k,c),y[:,None].expand(b,k).reshape(-1)).reshape(b,k)
            mean_loss=view_losses.mean()
            loss=0.35*global_loss+mean_loss

            if strategy in ("dual_tail","dual_anchor_tail"):
                tail=torch.topk(view_losses,k=3,dim=1).values.mean()
                loss=loss+0.45*tail

            if strategy=="dual_anchor_tail":
                conf=entropy_confidence(gl.detach())
                gl_expand=gl[:,None,:].expand_as(logits)
                div=symmetric_kl(logits,gl_expand).mean(1)
                anchor=(conf*div).mean()
                loss=loss+0.15*anchor

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()
            total += float(loss.item())*b
            n += b

        epoch_loss=total/max(n,1)
        if epoch_loss < best_loss - 1e-4:
            best_loss=epoch_loss
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            bad=0
        else:
            bad+=1
            if bad>=patience:
                break

    model.load_state_dict(best)
    return model

@torch.no_grad()
def evaluate(model, gx, lx, y, device):
    model.eval()
    gt=torch.tensor(gx,dtype=torch.float32,device=device)
    lt=torch.tensor(lx,dtype=torch.float32,device=device)
    _,logits=model(gt,lt)
    pred=logits.argmax(-1).cpu().numpy()
    zonings=[f"z{i}" for i in range(N_VIEWS)]

    correct=pred==y[:,None]
    exposed=correct.any(1)&(~correct).any(1)
    disagree=np.array([len(set(row.tolist()))>1 for row in pred])
    canon=correct[:,0]
    any_alt_wrong=(~correct[:,1:]).any(1)
    any_alt_correct=correct[:,1:].any(1)

    zacc=[accuracy_score(y,pred[:,i]) for i in range(N_VIEWS)]
    return {
        "canonical_accuracy":zacc[0],
        "canonical_balanced_accuracy":balanced_accuracy_score(y,pred[:,0]),
        "canonical_macro_f1":f1_score(y,pred[:,0],average="macro",zero_division=0),
        "mean_zoning_accuracy":float(np.mean(zacc)),
        "worst_zoning_accuracy":float(np.min(zacc)),
        "pier":float(exposed.mean()),
        "prediction_disagreement":float(disagree.mean()),
        "correct_to_wrong":float((canon&any_alt_wrong).mean()),
        "wrong_to_correct":float(((~canon)&any_alt_correct).mean()),
        "zoning_accuracies":dict(zip(zonings,zacc)),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",default="work/features")
    ap.add_argument("--out-dir",default="results/method_v2_pilot")
    args=ap.parse_args()
    set_seed()

    files=sorted(Path(args.input_dir).glob("*.parquet"))
    df=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
    cities=sorted(df.city.unique())
    rows=[]
    zoning_rows=[]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for test_city in cities:
        train_c=[c for c in cities if c!=test_city]
        _,trg,trl,try_=grouped_arrays(df,train_c)
        _,teg,tel,tey=grouped_arrays(df,[test_city])
        trg,trl,teg,tel=standardize(trg,trl,teg,tel)

        for strategy in ("dual_mean","dual_tail","dual_anchor_tail"):
            model=train_one(trg,trl,try_,strategy,device)
            m=evaluate(model,teg,tel,tey,device)
            rows.append({
                "city":test_city,
                "strategy":strategy,
                **{k:v for k,v in m.items() if k!="zoning_accuracies"},
            })
            for z,a in m["zoning_accuracies"].items():
                zoning_rows.append({"city":test_city,"strategy":strategy,"zoning":z,"accuracy":a})

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    s=pd.DataFrame(rows)
    s.to_csv(out/"city_summary.csv",index=False)
    pd.DataFrame(zoning_rows).to_csv(out/"per_zoning_accuracy.csv",index=False)
    agg=(s.groupby("strategy",as_index=False)
         .agg(mean_accuracy=("canonical_accuracy","mean"),
              mean_balanced_accuracy=("canonical_balanced_accuracy","mean"),
              mean_macro_f1=("canonical_macro_f1","mean"),
              mean_pier=("pier","mean"),
              mean_disagreement=("prediction_disagreement","mean"),
              mean_correct_to_wrong=("correct_to_wrong","mean"),
              mean_wrong_to_correct=("wrong_to_correct","mean"),
              min_city_accuracy=("canonical_accuracy","min")))
    agg.to_csv(out/"aggregate_summary.csv",index=False)
    (out/"run_info.json").write_text(json.dumps({"device":str(device),"seed":SEED},indent=2))

if __name__=="__main__":
    main()
