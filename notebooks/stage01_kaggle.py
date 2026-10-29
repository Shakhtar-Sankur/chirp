# chirp stage 0.1 on Kaggle: stage 0's blind search with gating and the chi-square veto.
# Notebook settings: Accelerator "GPU T4 x2", Internet on. Paste this file into one cell, then
# Save Version -> Save & Run All. About 1.5 hours; results print at the end and are saved as JSON.
import json
import os
import subprocess
import sys
import time

W = "/kaggle/working"
os.chdir(W)


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


sh("rm -rf chirp && git clone -q https://github.com/Shakhtar-Sankur/chirp")
sh(f"{sys.executable} -m pip install -q lalsuite h5py certifi")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("cd chirp && git log -1 --format=%h")
sys.path.insert(0, f"{W}/chirp")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from chirp import bank, data, gpu, search, veto  # noqa: E402

FS, T, F_LOW, P = 4096.0, 256.0, 20.0, 16
N = int(T * FS)
DF = 1.0 / T
f = np.fft.rfftfreq(N, 1.0 / FS)
EVENT_AT = T - 64.0           # each event sits 64 s before the end of its 256 s segment
VALID = (int(64 * FS), int((T - 4) * FS))   # samples not corrupted by template wrap-around
NEAR = (int((EVENT_AT - 1.0) * FS), int((EVENT_AT + 1.0) * FS))
L = int(FS)                   # 1 s clusters, the first starting half a second in, so the event is mid-cluster
FIRST = VALID[0] + L // 2
COUNT = (VALID[1] - FIRST) // L
WINDOW = 0.012
CACHE = f"{W}/gwcache"
dev = torch.device("cuda")

events = [e for e in json.load(open(f"{W}/chirp/data/gwtc4_o4a_events.json"))["events"]
          if {"H1", "L1"} <= set(e["detectors"]) and e["chirp_mass_det"] and 5.0 <= e["chirp_mass_det"] <= 200.0]
templates = json.load(open(f"{W}/chirp/data/bank_mc5_200.json"))["templates"]
print(f"{len(events)} two-detector events with detector-frame chirp mass 5-200; {len(templates)} templates", flush=True)

# 1. Data: find glitches in the whitened strain, gate them, then estimate the spectrum from gated data
prep, gates_log, t0 = [], {}, time.time()
for e in events:
    try:
        spec, glog = {}, {}
        for det in ("H1", "L1"):
            h, ts, fs = data.strain_around("O4a", det, e["gps"], 600, 200, CACHE)
            if np.isnan(h).any():
                raise ValueError(f"{det} has gaps")
            pf, psd = search.welch_psd(h, FS)
            gates = veto.find_gates(h, FS, pf, psd, F_LOW)
            h = veto.apply_gates(h, FS, gates)
            glog[det] = [round(g + ts - e["gps"], 3) for g in gates]   # seconds from the event
            pf, psd = search.welch_psd(h, FS)
            inv = search.inverse_psd(pf, psd, f, FS, F_LOW)
            i0 = int(round((e["gps"] - EVENT_AT - ts) * FS))
            seg = h[i0:i0 + N] * search.tukey(N, 0.1)
            spec[det] = gpu.to_device(np.fft.rfft(seg) / FS, inv, (f >= F_LOW) & (f <= 512.0), dev)
        prep.append((e, spec))
        gates_log[e["name"]] = glog
    except Exception as ex:  # a gap or a missing file: report it and move on
        print(f"skip {e['name']}: {ex}", flush=True)
    for p in os.listdir(CACHE) if os.path.isdir(CACHE) else []:
        os.remove(os.path.join(CACHE, p))
print(f"prepared {len(prep)} events in {(time.time() - t0) / 60:.1f} min", flush=True)

# 2. Search: the loudest template in every 1 s cluster, and the loudest near the catalog time
cl = {e["name"]: [torch.zeros(COUNT, device=dev), torch.full((COUNT,), -1, device=dev, dtype=torch.long),
                  torch.zeros(COUNT, device=dev, dtype=torch.long)] for e, _ in prep}
near = {e["name"]: (0.0, -1, -1) for e, _ in prep}
B, t0 = 48, time.time()
for b0 in range(0, len(templates), B):
    batch = templates[b0:b0 + B]
    hs = gpu.templates_to_device([bank.imr_template(t[0], t[1], f, F_LOW) for t in batch], dev)
    for e, spec in prep:
        cv, ct, ci, nv, ni = gpu.batch_network_snr_clusters(spec["H1"][0], spec["L1"][0], spec["H1"][1], spec["L1"][1],
                                                            hs, DF, FS, WINDOW, FIRST, L, COUNT, NEAR)
        c = cl[e["name"]]
        up = cv > c[0]
        c[0] = torch.where(up, cv, c[0])
        c[1] = torch.where(up, ct + b0, c[1])
        c[2] = torch.where(up, ci, c[2])
        j = int(torch.argmax(nv))
        if float(nv[j]) > near[e["name"]][0]:
            near[e["name"]] = (float(nv[j]), b0 + j, int(ni[j]))
    if (b0 // B) % 10 == 0:
        print(f"  templates {b0 + len(batch)}/{len(templates)}  {(time.time() - t0) / 60:.1f} min", flush=True)

# 3. Chi-square pass: rescore every cluster's template, and the near template, with reweighted SNR
need = {}
for e, _ in prep:
    n = e["name"]
    c = [x.cpu().numpy() for x in cl[n]]
    cl[n] = c
    for k in range(COUNT):
        need.setdefault(int(c[1][k]), []).append((n, k))
    need.setdefault(near[n][1], []).append((n, -1))
spec_of = {e["name"]: s for e, s in prep}
hat = {e["name"]: np.zeros(COUNT) for e, _ in prep}
hat_i = {e["name"]: np.zeros(COUNT, dtype=int) for e, _ in prep}
near_hat = {}
t0 = time.time()
for m, ti in enumerate(sorted(need)):
    h = gpu.templates_to_device([bank.imr_template(templates[ti][0], templates[ti][1], f, F_LOW)], dev)[0]
    for n in sorted({x[0] for x in need[ti]}):
        s = spec_of[n]
        _, rhat = gpu.rescore_network(s["H1"][0], s["L1"][0], s["H1"][1], s["L1"][1], h, DF, FS, WINDOW, P)
        for _, k in (x for x in need[ti] if x[0] == n):
            if k < 0:
                v, i = rhat[NEAR[0]:NEAR[1]].max(dim=0)
                near_hat[n] = (float(v), int(i) + NEAR[0])
            else:
                a = FIRST + k * L
                v, i = rhat[a:a + L].max(dim=0)
                hat[n][k], hat_i[n][k] = float(v), int(i) + a
    if m % 200 == 0:
        print(f"  chi-square pass {m + 1}/{len(need)} templates  {(time.time() - t0) / 60:.1f} min", flush=True)

# 4. Report against the catalog: blind before (gating only) and after the chi-square reweighting
rows = []
for e, _ in prep:
    n = e["name"]
    c = cl[n]
    kb = int(np.argmax(c[0]))
    kh = int(np.argmax(hat[n]))
    nn, tn, i_n = near[n]
    row = {"name": n, "catalog_snr": e["snr_search_max"], "catalog_mc_det": e["chirp_mass_det"],
           "near_snr": nn, "near_dt_ms": (i_n / FS - EVENT_AT) * 1e3, "near_mc": templates[tn][2],
           "near_newsnr": near_hat[n][0],
           "gated_blind_snr": float(c[0][kb]), "gated_blind_dt_s": int(c[2][kb]) / FS - EVENT_AT,
           "blind_newsnr": float(hat[n][kh]), "blind_newsnr_dt_s": int(hat_i[n][kh]) / FS - EVENT_AT,
           "blind_newsnr_mc": templates[int(c[1][kh])][2], "gates_s_from_event": gates_log[n]}
    row["gated_blind_is_event"] = abs(row["gated_blind_dt_s"]) < 0.1
    row["blind_is_event"] = abs(row["blind_newsnr_dt_s"]) < 0.1
    srt = np.sort(hat[n])[::-1]
    row["next_loudest_newsnr"] = float(srt[1]) if row["blind_is_event"] else float(srt[0])
    rows.append(row)
rows.sort(key=lambda x: -x["catalog_snr"])
json.dump({"fs": FS, "segment_s": T, "f_low": F_LOW, "chisq_bins": P, "templates": len(templates), "events": rows},
          open(f"{W}/stage01_o4a.json", "w"), indent=1)
print("\n=== CHIRP STAGE 0.1 SUMMARY (copy this) ===")
print(f"{'event':18s} {'catalog':>7s} {'snr':>5s} {'new':>5s}  {'gated blind':>18s}  {'after chi-square':>22s}  gates")
for x in rows:
    ng = sum(len(v) for v in x["gates_s_from_event"].values())
    print(f"{x['name']:18s} {x['catalog_snr']:7.1f} {x['near_snr']:5.1f} {x['near_newsnr']:5.1f}  "
          f"{'EVENT' if x['gated_blind_is_event'] else 'other'} {x['gated_blind_snr']:6.1f} {x['gated_blind_dt_s']:+6.1f} s  "
          f"{'EVENT' if x['blind_is_event'] else 'other'} {x['blind_newsnr']:6.1f} {x['blind_newsnr_dt_s']:+6.1f} s  "
          f"{ng}")
g = sum(x["gated_blind_is_event"] for x in rows)
ok = sum(x["blind_is_event"] for x in rows)
print(f"\nloudest trigger in 188 s is the catalog event (within 0.1 s): gating only {g} of {len(rows)}; "
      f"gating + chi-square {ok} of {len(rows)}")
print(f"median reweighted / plain SNR at the events: {np.median([x['near_newsnr'] / x['near_snr'] for x in rows]):.3f}")
close = [x["name"] for x in rows if any(abs(t) < 1.0 for v in x["gates_s_from_event"].values() for t in v)]
print(f"events with a gate within 1 s: {close if close else 'none'}")
