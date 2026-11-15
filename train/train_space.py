#!/usr/bin/env python3
import os, re, json, argparse, warnings, time
from collections import Counter
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ["SPACE_DATA"])
ap.add_argument("--cache", default="cell3_cache_v5.npz")
ap.add_argument("--meta", default=None)
ap.add_argument("--records", default="dataset_records_final.json")
ap.add_argument("--out-suffix", default="")
ap.add_argument("--seed-list", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--p1-epochs", type=int, default=10)
ap.add_argument("--p2-epochs", type=int, default=12)
ap.add_argument("--p3-epochs", type=int, default=150)
ap.add_argument("--early-stop", type=int, default=40)
ap.add_argument("--target-active", type=int, default=512)
ap.add_argument("--lam-damp", type=float, default=0.25)
ap.add_argument("--lam-max", type=float, default=1e6)
ap.add_argument("--fa-cap", type=float, default=0.107)
ap.add_argument("--pareto-deltas", default="0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99")
ap.add_argument("--pareto-boot", type=int, default=500)
ap.add_argument("--no-half", dest="half", action="store_false")
ap.add_argument("--no-preload", action="store_true")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()

SAVE_DIR   = os.path.join(args.data, "merged_output")
CACHE_NPZ  = os.path.join(SAVE_DIR, args.cache)
CACHE_META = os.path.join(SAVE_DIR, args.meta or args.cache.replace("cell3_cache", "cell3_meta").replace(".npz", ".json"))
RECORDS    = os.path.join(SAVE_DIR, args.records)
SPLIT_PATH = os.path.join(args.data, "baseline_outputs", "train_test_split.json")
ANCHORS    = os.path.join(SAVE_DIR, "policy_kb", "policy_anchors_cat.pt")
OUT_DIR    = os.path.join(SAVE_DIR, "main_softmax" + args.out_suffix); os.makedirs(OUT_DIR, exist_ok=True)
print(f"[cache]   {CACHE_NPZ}\n[meta]    {CACHE_META}\n[records] {RECORDS}\n[output]  {OUT_DIR}")
for _p in (CACHE_NPZ, CACHE_META, RECORDS, SPLIT_PATH, ANCHORS):
    if not os.path.exists(_p): raise SystemExit(f"Missing file: {_p}")
device = torch.device(args.device)
DROP = {3, 4}
TOX = [1, 2]
N_CLASSES = 5
DEAD_COLS = [3, 4]
VNAME = "full"

CFG = dict(lr_evidence=5e-4, lr_gate=0.1, lr_projector=1e-3, weight_decay=1e-2, grad_clip=1.0,
           lambda_seg=0.5, lambda_intra=0.2, lambda_cross=0.05, lambda_align=0.1,
           temp_cross=0.2, temp_policy=0.1, lambda_sparse_init=5e-3, ramp_fraction=0.3)

cache = np.load(CACHE_NPZ, allow_pickle=False)
Z, ycls, mmc = cache["Z_frames"], cache["y_class"], cache["masks_mc"]
N, T, D = Z.shape
filenames = json.load(open(CACHE_META))["filenames"]
test_vid = {str(v) for v in json.load(open(SPLIT_PATH))["test_video_ids"]}
fn2vid = {}
for r in json.load(open(RECORDS)):
    fn = r.get("filename") or r.get("file") or ""
    vid = r.get("video_id") or r.get("id")
    if fn and vid is not None: fn2vid[fn] = str(vid); fn2vid[fn[:-4]] = str(vid)
_ACCT_RE = re.compile(r"(@.+?)_video_\d+")
ACCT = np.array([(_ACCT_RE.match(fn).group(1) if _ACCT_RE.match(fn) else fn)
                 for fn in filenames])
_miss = int(sum(1 for fn in filenames if not _ACCT_RE.match(fn)))
print(f"[accounts] {len(set(ACCT))} groups / {len(filenames)} videos"
      + (f"   !! {_miss} filenames did not match the account pattern; each forms its own group" if _miss else "   all matched"))
tr_i, te_i = [], []
for i, fn in enumerate(filenames):
    if int(ycls[i]) in DROP: continue
    v = fn2vid.get(fn) or fn2vid.get(fn[:-4])
    (te_i if (v and v in test_vid) else tr_i).append(i)
tr_i, te_i = np.array(tr_i), np.array(te_i)
ytr_np, yte_np, Mte_np = ycls[tr_i], ycls[te_i], mmc[te_i]
print(f"train={len(tr_i)} test={len(te_i)} harmful_test={int((yte_np!=0).sum())} D={D}")

FDT = torch.bfloat16 if args.half else torch.float32
PRELOAD = (device.type == "cuda") and not args.no_preload
if PRELOAD:
    need = (len(tr_i) + len(te_i)) * T * D * (2 if args.half else 4) / 1e9
    free, total = torch.cuda.mem_get_info()
    print(f"Preloading features to GPU: about {need:.1f} GB needed, {free/1e9:.1f} / {total/1e9:.1f} GB free")
    if need > free / 1e9 * 0.85:
        print("  Not enough GPU memory; sending features batch by batch instead"); PRELOAD = False
if PRELOAD:
    t0 = time.time()
    ZTR = torch.empty((len(tr_i), T, D), dtype=FDT, device=device)
    for s in range(0, len(tr_i), 128):
        ZTR[s:s+128] = torch.from_numpy(np.ascontiguousarray(Z[tr_i[s:s+128]])).to(device, FDT)
    ZTE = torch.empty((len(te_i), T, D), dtype=FDT, device=device)
    for s in range(0, len(te_i), 128):
        ZTE[s:s+128] = torch.from_numpy(np.ascontiguousarray(Z[te_i[s:s+128]])).to(device, FDT)
    print(f"  Done in {time.time()-t0:.1f}s")
    def train_feats(rows_local): return ZTR[rows_local].float() * SCALE
    def test_feats(sl):          return ZTE[sl].float() * SCALE
else:
    def train_feats(rows_local):
        return torch.from_numpy(np.ascontiguousarray(Z[tr_i[rows_local]])).float().to(device) * SCALE
    def test_feats(sl):
        return torch.from_numpy(np.ascontiguousarray(Z[te_i[sl]])).float().to(device) * SCALE

_acc, _n = 0.0, 0
for s_ in range(0, len(tr_i), 256):
    _b = np.abs(np.asarray(Z[tr_i[s_:s_+256]])); _acc += _b.sum(); _n += _b.size
SCALE = float(1.0 / (_acc / max(_n, 1) + 1e-12))
print(f"Input scaling (global): SCALE={SCALE:.4g}")

YTR = torch.from_numpy(ytr_np).long().to(device)
MTR = torch.from_numpy(mmc[tr_i]).long().to(device)

SKF = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                random_state=42).split(tr_i, ytr_np, groups=ACCT[tr_i]))


class GatedEvidenceModel(nn.Module):
    def __init__(self, n_features, n_classes=5):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.full((n_features,), 2.2))
        self.evidence = nn.Linear(n_features, n_classes)
        self.register_buffer("select_mask", torch.ones(n_features))
    def select_all(self):
        with torch.no_grad():
            self.select_mask.fill_(1.0)
        return int(self.select_mask.sum().item())
    def set_gate_threshold(self, delta):
        with torch.no_grad():
            self.select_mask.copy_((torch.sigmoid(self.gate_logits) > float(delta)).float())
        return int(self.select_mask.sum().item())
    def forward(self, z):
        g = torch.sigmoid(self.gate_logits) * self.select_mask
        zt = z * g.unsqueeze(0).unsqueeze(0)
        e = self.evidence(zt)
        return e, zt, g
    def get_gated_video_repr(self, z):
        zt = z * (torch.sigmoid(self.gate_logits) * self.select_mask).unsqueeze(0).unsqueeze(0)
        return zt.amax(1)
    def get_active_features(self, th=0.5):
        with torch.no_grad():
            return int(((torch.sigmoid(self.gate_logits) > th) & (self.select_mask > 0)).sum().item())


class PolicyProjector(nn.Module):
    def __init__(self, sae_dim, clip_dim=768):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(sae_dim, 1024), nn.ReLU(), nn.Linear(1024, clip_dim))
    def forward(self, z): return F.normalize(self.projector(z), dim=-1)


def loss_seg_multiclass(e, m, cw=None):
    B, Tt, C = e.shape
    return F.cross_entropy(e.reshape(B*Tt, C), m.reshape(B*Tt), weight=cw)


def loss_intra_pairwise_multiclass(e, m):
    harm = e[:, :, TOX].max(dim=-1).values - e[:, :, 0]
    pos, neg = (m > 0), (m == 0)
    el = (pos.sum(1) > 0) & (neg.sum(1) > 0)
    if not el.any(): return torch.zeros((), device=e.device, requires_grad=True)
    s, p, n = harm[el], pos[el], neg[el]
    diff = s.unsqueeze(2) - s.unsqueeze(1)
    valid = p.unsqueeze(2) & n.unsqueeze(1)
    lm = F.softplus(-diff) * valid.float()
    return (lm.sum((1, 2)) / valid.sum((1, 2)).float().clamp(min=1)).mean()


def loss_cross_video_multiclass(zt, y, m, temperature=0.2):
    B = zt.shape[0]
    hm = (m > 0).float().unsqueeze(-1)
    harm_rep = torch.where(hm > 0, zt, torch.full_like(zt, -1e9)).amax(1)
    all_rep = zt.amax(1)
    use_harm = ((y > 0) & ((m > 0).any(1))).unsqueeze(-1).float()
    reps = F.normalize(use_harm * harm_rep + (1 - use_harm) * all_rep, dim=-1)
    sim = (reps @ reps.T) / temperature
    self_m = torch.eye(B, device=sim.device).bool()
    sim = sim.masked_fill(self_m, -1e9)
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & (~self_m)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has = pos.any(1)
    if not has.any(): return torch.zeros((), device=zt.device, requires_grad=True)
    return -((logp * pos.float()).sum(1)[has] / pos.float().sum(1)[has]).mean()


def loss_policy_align(proj, model, zb, y, anchors, temperature=0.1):
    v = proj(model.get_gated_video_repr(zb))
    sim = (v @ anchors.T) / temperature
    return F.cross_entropy(sim, y)


def class_weights(y):
    c = Counter(int(v) for v in y); tot = len(y)
    w = torch.tensor([tot/(N_CLASSES*c[i]) if c[i] > 0 else 1.0 for i in range(N_CLASSES)], dtype=torch.float32)
    return (w / w.mean()).to(device)


@torch.no_grad()
def _val_predict(model, rows_val):
    model.eval(); P = []
    for s_ in range(0, len(rows_val), 256):
        z = train_feats(rows_val[s_:s_+256]); e, _, _ = model(z)
        _v = e.max(1)[0]
        _v[:, DEAD_COLS] = -1e9
        P.append(_v.argmax(-1).cpu().numpy()); del z, e, _v
    model.train()
    return ytr_np[rows_val], np.concatenate(P)


def _val_metric(yv, p):
    tm = yv > 0
    if tm.sum() == 0: return None
    nb = int((yv == 0).sum())
    fa = (((yv == 0) & (p > 0)).sum() / nb) if nb else 0.0
    sc = float(f1_score(yv[tm], p[tm], average="macro", labels=TOX, zero_division=0))
    return sc if fa <= args.fa_cap else sc * float(args.fa_cap / max(fa, 1e-9))


def val_score(model, rows_val):
    return _val_metric(*_val_predict(model, rows_val))


PARETO_DELTAS = [float(x) for x in args.pareto_deltas.split(",") if x.strip()]


def _boot_se(yv, p, g, n_boot):
    U = np.unique(g)
    if len(U) < 2: return 0.0
    rng = np.random.default_rng(0)
    ib = {u: np.where(g == u)[0] for u in U}
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(U, len(U), replace=True)
        idx = np.concatenate([ib[u] for u in pick])
        m = _val_metric(yv[idx], p[idx])
        if m is not None: bs.append(m)
    return float(np.std(bs, ddof=1)) if len(bs) > 1 else 0.0


def pareto_front(cands):
    seen, uniq = set(), []
    for c in cands:
        if c["n"] not in seen:
            seen.add(c["n"]); uniq.append(c)
    front = [x for x in uniq
             if not any(y is not x and y["sc"] >= x["sc"] and y["n"] <= x["n"]
                        and (y["sc"] > x["sc"] or y["n"] < x["n"]) for y in uniq)]
    return sorted(front, key=lambda c: c["n"])


def pareto_select(model, rows_val, groups_val):
    cands = []
    for d in PARETO_DELTAS:
        n = model.set_gate_threshold(d)
        yv, p = _val_predict(model, rows_val)
        cands.append(dict(tag=float(d), n=n, na=model.get_active_features(),
                          sc=_val_metric(yv, p), yv=yv, p=p))
    valid = [c for c in cands if c["sc"] is not None]
    if not valid:
        n = model.select_all()
        return 0.0, n, None, []
    front = pareto_front(valid)
    best = max(front, key=lambda c: c["sc"])
    se = _boot_se(best["yv"], best["p"], groups_val, args.pareto_boot)
    thr = best["sc"] - se
    pick = min([c for c in front if c["sc"] >= thr], key=lambda c: c["n"])
    print(f"      [pareto] {len(valid)} candidates -> {len(front)} on the front   "
          + "  ".join(f"({c['tag']:.2f}, C={c['n']}, P={c['sc']:.4f})" for c in front))
    print(f"      [pareto] best on front P={best['sc']:.4f} @C={best['n']}  SE={se:.4f}  "
          f"threshold={thr:.4f}  ->  selected delta={pick['tag']:.2f}  C={pick['n']}  P={pick['sc']:.4f}  "
          f"active_gates={pick['na']}")
    model.set_gate_threshold(pick["tag"])
    return pick["tag"], pick["n"], pick["sc"], [
        dict(tag=c["tag"], C=int(c["n"]), P=float(c["sc"]), n_active=int(c["na"]))
        for c in front]


def train_space(rows_local, rows_val, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GatedEvidenceModel(D, N_CLASSES).to(device)
    proj = PolicyProjector(D).to(device)
    anchors = torch.load(ANCHORS).to(device)
    cw = class_weights(ytr_np[rows_local])
    n = len(rows_local)

    def run(epochs, use, opt, lam_sparse=None, track=False):
        best_s, best_state, bad = -1.0, None, 0
        lam_cur = lam_sparse
        for ep in range(epochs):
            ramp = min(1.0, (ep + 1) / max(1, int(epochs * CFG["ramp_fraction"])))
            perm = np.random.permutation(n)
            for s in range(0, n, args.batch):
                b = rows_local[perm[s:s+args.batch]]
                z = train_feats(b); y = YTR[b]; m = MTR[b]
                e, zt, _ = model(z)
                loss = F.cross_entropy(e.max(1)[0], y, weight=cw)
                if use.get("seg"):
                    loss = loss + CFG["lambda_seg"] * loss_seg_multiclass(e, m, cw)
                if use.get("intra"):
                    loss = loss + CFG["lambda_intra"] * ramp * loss_intra_pairwise_multiclass(e, m)
                if use.get("cross"):
                    loss = loss + CFG["lambda_cross"] * ramp * loss_cross_video_multiclass(zt, y, m, CFG["temp_cross"])
                if use.get("policy"):
                    loss = loss + CFG["lambda_align"] * ramp * loss_policy_align(
                        proj, model, z, y, anchors, CFG["temp_policy"])
                if lam_cur is not None:
                    loss = loss + lam_cur * torch.sigmoid(model.gate_logits).sum() / D
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for gp in opt.param_groups for p in gp["params"]], CFG["grad_clip"])
                opt.step()
                del z, y, m, e, zt, loss
            if lam_cur is not None and args.target_active > 0:
                with torch.no_grad():
                    n_act = int((torch.sigmoid(model.gate_logits) > 0.5).sum().item())
                ratio = max(n_act, 1) / max(args.target_active, 1)
                lam_cur = float(np.clip(lam_cur * (ratio ** args.lam_damp), 1e-8, args.lam_max))
                if (ep + 1) % 5 == 0 and ep < 30 or (ep + 1) % 25 == 0:
                    _ok = (args.target_active * 0.5) <= n_act <= (args.target_active * 2)
                    print(f"      ep{ep+1:>3} n_active={n_act:>7} lam={lam_cur:.4g} "
                          f"sparse_ok={_ok}", flush=True)
            if lam_cur is not None and args.target_active > 0:
                _na = int((torch.sigmoid(model.gate_logits) > 0.5).sum().item())
                _sparse_ok = (args.target_active * 0.5) <= _na <= (args.target_active * 2)
            else:
                _sparse_ok = True
            if track and args.early_stop > 0 and (ep + 1) % 5 == 0:
                sc = val_score(model, rows_val)
                if sc is not None:
                    if sc > best_s:
                        best_s, bad = sc, 0
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    else:
                        bad += 5
                        if _sparse_ok and bad >= args.early_stop:
                            if best_state is not None: model.load_state_dict(best_state)
                            return ep + 1, best_s
        if track and best_state is not None: model.load_state_dict(best_state)
        return epochs, best_s

    ev = lambda: [{"params": model.evidence.parameters(), "lr": CFG["lr_evidence"], "weight_decay": CFG["weight_decay"]}]
    model.gate_logits.requires_grad_(False)
    run(args.p1_epochs, dict(seg=True), torch.optim.AdamW(ev()))
    run(args.p2_epochs, dict(seg=True, intra=True, cross=True), torch.optim.AdamW(ev()))
    model.gate_logits.requires_grad_(True)
    ep3, bs3 = run(args.p3_epochs, dict(seg=True, intra=True, cross=True, policy=True),
        torch.optim.AdamW(ev() + [{"params": [model.gate_logits], "lr": CFG["lr_gate"], "weight_decay": 0.0},
                                  {"params": proj.parameters(), "lr": CFG["lr_projector"], "weight_decay": CFG["weight_decay"]}]),
        lam_sparse=CFG["lambda_sparse_init"], track=True)
    ptag, nfeat, vk, front = pareto_select(model, rows_val, ACCT[tr_i][rows_val])
    return model, ep3, (vk if vk is not None else bs3), ptag, nfeat, front, proj, anchors


ANCHOR_DOC   = [1, 2]
ANCHOR_PLUS0 = [0, 1, 2]
ANCHOR_ALL   = [0, 1, 2, 3, 4]


@torch.no_grad()
def evaluate(model, proj, anchors):
    R = np.arange(len(te_i))
    model.eval(); P, S, PR, HS, VP = [], [], [], [], []
    for s in range(0, len(R), 64):
        z = test_feats(R[s:s+64])
        e, _, _ = model(z)
        v = e.max(1)[0]
        v[:, DEAD_COLS] = -1e9
        P.append(v.argmax(-1).cpu().numpy())
        PR.append(F.softmax(v, -1).cpu().numpy())
        S.append((1 - F.softmax(e, -1)[..., 0]).cpu().numpy())
        HS.append((e[:, :, TOX].max(dim=-1).values - e[:, :, 0]).cpu().numpy())
        VP.append((proj(model.get_gated_video_repr(z)) @ anchors.T).cpu().numpy())
        del z, e, v
    p, sg = np.concatenate(P), np.concatenate(S)
    hs = np.concatenate(HS)
    proba = np.concatenate(PR)
    yte_np_r, Mte_np_r = yte_np[R], Mte_np[R]
    yb, pb = (yte_np_r != 0).astype(int), (p != 0).astype(int)
    tm = yte_np_r > 0
    out = dict(harm_f1=f1_score(yte_np_r, p, average="macro", labels=TOX, zero_division=0),
               toxic_only=(f1_score(yte_np_r[tm], p[tm], average="macro", labels=TOX,
                                    zero_division=0) if tm.any() else float("nan")),
               false_alarm=int(((yte_np_r == 0) & (p > 0)).sum()),
               n_benign=int((yte_np_r == 0).sum()),
               harm_p=precision_score(yte_np_r, p, average="macro", labels=TOX, zero_division=0),
               harm_r=recall_score(yte_np_r, p, average="macro", labels=TOX, zero_division=0),
               bin_f1=f1_score(yb, pb, average="macro", zero_division=0),
               n_active=model.get_active_features())
    for c in TOX:
        out[f"f1_c{c}"] = f1_score(yte_np_r, p, labels=[c], average="macro", zero_division=0)
    sel = yte_np_r != 0
    Sg, M = sg[sel], (Mte_np_r[sel] > 0).astype(int)
    keep = (M.sum(1) > 0) & (M.sum(1) < M.shape[1]); Sg, M = Sg[keep], M[keep]
    if len(Sg) == 0:
        out.update(pair_rank=float("nan"), strict_rank=float("nan"))
    else:
        ok = tot = strict = 0
        for i in range(len(Sg)):
            pos, neg = Sg[i][M[i] == 1], Sg[i][M[i] == 0]
            c = pos[:, None] - neg[None, :]
            ok += (c > 0).sum() + 0.5*(c == 0).sum(); tot += c.size
            strict += int(pos.min() > neg.max())
        out.update(pair_rank=ok/tot, strict_rank=strict/len(Sg))

    _hs_t = hs[sel]
    _sy   = (Mte_np_r[sel] > 0).astype(int)
    if len(_hs_t) == 0:
        out.update(top1_hit_toxic=float("nan"), top1_hit_eligible=float("nan"),
                   n_toxic=0, n_eligible=0)
    else:
        _hit = np.array([_sy[i, int(_hs_t[i].argmax())] > 0 for i in range(len(_hs_t))])
        _el  = (_sy.sum(1) > 0) & (_sy.sum(1) < _sy.shape[1])
        out.update(top1_hit_toxic=float(_hit.mean()),
                   top1_hit_eligible=(float(_hit[_el].mean()) if _el.any() else float("nan")),
                   n_toxic=int(len(_hs_t)), n_eligible=int(_el.sum()))
    sim = np.concatenate(VP)
    _tox = np.isin(yte_np_r, TOX)
    for _nm, _keep in (("anchor_doc", ANCHOR_DOC), ("anchor_plus_safe", ANCHOR_PLUS0),
                       ("anchor_raw_all5", ANCHOR_ALL)):
        _k = [c for c in _keep if c < sim.shape[1]]
        if not _k: continue
        _pred = np.asarray(_k)[sim[:, _k].argmax(axis=1)]
        out[f"policy_{_nm}_harmful"] = float((_pred[_tox] == yte_np_r[_tox]).mean()) if _tox.any() else float("nan")
        out[f"policy_{_nm}_all"] = float((_pred == yte_np_r).mean())
    _pa = np.asarray(ANCHOR_ALL)[sim[:, ANCHOR_ALL].argmax(axis=1)] if sim.shape[1] >= 5 else None
    if _pa is not None and _tox.any():
        out["dead_anchor_won_harmful"] = float(np.isin(_pa[_tox], [3, 4]).mean())
        out["safe_anchor_won_harmful"] = float((_pa[_tox] == 0).mean())
    model.train()
    out["_proba"] = proba
    return out


RES = os.path.join(OUT_DIR, "retrain_results.json")
res = json.load(open(RES)) if os.path.exists(RES) else {}
SEED_LIST = [int(x) for x in args.seed_list.split(",") if x.strip()]
print(f"device={device}  folds={args.folds}  seeds={SEED_LIST}")
t0 = time.time(); done_n = 0
total = len(SKF) * len(SEED_LIST)
res.setdefault(VNAME, [])
done = {(r["fold"], r["seed"]) for r in res[VNAME]}
for fi, (tr_sub, va_sub) in enumerate(SKF, 1):
    for seed in SEED_LIST:
        if (fi, seed) in done: done_n += 1; continue
        ts = time.time()
        m, ep_used, val_best, ptag, nfeat, front, _proj, _anch = train_space(tr_sub, va_sub, seed=seed*100+fi)
        r = evaluate(m, _proj, _anch)
        pr = r.pop("_proba")
        np.savez_compressed(os.path.join(OUT_DIR, f"proba_{VNAME}_f{fi}_s{seed}.npz"),
                            proba=pr.astype(np.float32),
                            video_ids=np.array([fn2vid.get(filenames[i], "") for i in te_i]),
                            y_true=yte_np.astype(np.int64))
        np.savez_compressed(
            os.path.join(OUT_DIR, f"params_{VNAME}_f{fi}_s{seed}.npz"),
            gate=torch.sigmoid(m.gate_logits).detach().cpu().numpy().astype(np.float32),
            select_mask=m.select_mask.detach().cpu().numpy().astype(np.float32),
            W=m.evidence.weight.detach().cpu().numpy().astype(np.float32),
            b=m.evidence.bias.detach().cpu().numpy().astype(np.float32))
        r.update(fold=fi, seed=seed, ep_used=int(ep_used),
                 val_best=(float(val_best) if val_best is not None else None),
                 pareto_tag=float(ptag), pareto_C=int(nfeat), pareto_front=front)
        res[VNAME].append(r); del m
        if device.type == "cuda": torch.cuda.empty_cache()
        json.dump(res, open(RES, "w"), indent=1, default=float)
        done_n += 1
        eta = (time.time()-t0)/max(done_n,1)*(total-done_n)/60
        _fa = r['false_alarm'] / max(r['n_benign'], 1)
        print(f"  fold{fi} seed{seed}  HarmF1={r['harm_f1']:.4f} "
              f"ToxicOnly={r['toxic_only']:.4f} FA={_fa:.1%} |S|={r['n_active']} "
              f"ep={r['ep_used']}  ({time.time()-ts:.0f}s, about {eta:.0f} min left)")
print(f"\nAll done in {(time.time()-t0)/60:.1f} min. Results: {RES}")
