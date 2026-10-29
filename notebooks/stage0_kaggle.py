# chirp stage 0 on Kaggle: re-find the GWTC-4.0 O4a events with an independent GPU search.
# Notebook settings: Accelerator "GPU T4 x2", Internet on. Paste this file into one cell, then
# Save Version -> Save & Run All. About an hour; results print at the end and are saved as JSON.
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
sys.path.insert(0, f"{W}/chirp")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from chirp import bank, data, gpu, search  # noqa: E402

FS, T, F_LOW = 4096.0, 256.0, 20.0
N = int(T * FS)
DF = 1.0 / T
f = np.fft.rfftfreq(N, 1.0 / FS)
EVENT_AT = T - 64.0           # each event sits 64 s before the end of its 256 s segment
VALID = (int(64 * FS), int((T - 4) * FS))   # samples not corrupted by template wrap-around
NEAR = (int((EVENT_AT - 1.0) * FS), int((EVENT_AT + 1.0) * FS))
CACHE = f"{W}/gwcache"
dev = torch.device("cuda")

events = [e for e in json.load(open(f"{W}/chirp/data/gwtc4_o4a_events.json"))["events"]
          if {"H1", "L1"} <= set(e["detectors"]) and e["chirp_mass_det"] and 5.0 <= e["chirp_mass_det"] <= 200.0]
templates = json.load(open(f"{W}/chirp/data/bank_mc5_200.json"))["templates"]
print(f"{len(events)} two-detector events with detector-frame chirp mass 5-200; {len(templates)} templates", flush=True)

# 1. Data: whitening inputs for every event, then the 4096 s files are deleted to save disk
prep, t0 = [], time.time()
for e in events:
    try:
        spec = {}
        for det in ("H1", "L1"):
            h, ts, fs = data.strain_around("O4a", det, e["gps"], 600, 200, CACHE)
            if np.isnan(h).any():
                raise ValueError(f"{det} has gaps")
            pf, psd = search.welch_psd(h, FS)
            inv = search.inverse_psd(pf, psd, f, FS, F_LOW)
            i0 = int(round((e["gps"] - EVENT_AT - ts) * FS))
            seg = h[i0:i0 + N] * search.tukey(N, 0.1)
            spec[det] = gpu.to_device(np.fft.rfft(seg) / FS, inv, (f >= F_LOW) & (f <= 512.0), dev)
        prep.append((e, spec))
    except Exception as ex:  # a gap or a missing file: report it and move on
        print(f"skip {e['name']}: {ex}", flush=True)
    for p in os.listdir(CACHE) if os.path.isdir(CACHE) else []:
        os.remove(os.path.join(CACHE, p))
print(f"prepared {len(prep)} events in {(time.time() - t0) / 60:.1f} min", flush=True)

# 2. Search: each template batch is generated once and filtered against every event
best = {e["name"]: {"blind": (0, -1, -1), "near": (0, -1, -1), "single": (0, 0)} for e, _ in prep}
B, t0 = 48, time.time()
for b0 in range(0, len(templates), B):
    batch = templates[b0:b0 + B]
    hs = gpu.templates_to_device([bank.imr_template(t[0], t[1], f, F_LOW) for t in batch], dev)
    for e, spec in prep:
        out = gpu.batch_network_snr(spec["H1"][0], spec["L1"][0], spec["H1"][1], spec["L1"][1], hs, DF, FS,
                                    0.012, VALID, NEAR)
        bv, bi, nv, ni, rh, rl = (x.cpu().numpy() for x in out)
        r = best[e["name"]]
        j = int(np.argmax(bv))
        if bv[j] > r["blind"][0]:
            r["blind"] = (float(bv[j]), b0 + j, int(bi[j]))
        j = int(np.argmax(nv))
        if nv[j] > r["near"][0]:
            r["near"] = (float(nv[j]), b0 + j, int(ni[j]))
            r["single"] = (float(rh[j]), float(rl[j]))
    if (b0 // B) % 5 == 0:
        print(f"  templates {b0 + len(batch)}/{len(templates)}  {(time.time() - t0) / 60:.1f} min", flush=True)

# 3. Report against the catalog
rows = []
for e, _ in prep:
    r = best[e["name"]]
    nb, tb, ib = r["blind"]
    nn, tn, i_n = r["near"]
    row = {"name": e["name"], "catalog_snr": e["snr_search_max"], "catalog_mc_det": e["chirp_mass_det"],
           "near_snr": nn, "near_dt_ms": (i_n / FS - EVENT_AT) * 1e3, "near_mc": templates[tn][2],
           "near_rho_H1": r["single"][0], "near_rho_L1": r["single"][1],
           "blind_snr": nb, "blind_dt_s": ib / FS - EVENT_AT, "blind_mc": templates[tb][2]}
    row["blind_is_event"] = abs(row["blind_dt_s"]) < 0.05
    rows.append(row)
rows.sort(key=lambda x: -x["catalog_snr"])
json.dump({"fs": FS, "segment_s": T, "f_low": F_LOW, "templates": len(templates), "events": rows},
          open(f"{W}/stage0_o4a.json", "w"), indent=1)
print("\n=== CHIRP STAGE 0 SUMMARY (copy this) ===")
print(f"{'event':18s} {'catalog':>7s} {'chirp':>6s} {'ratio':>6s} {'dt ms':>6s}  {'Mc cat':>6s} {'Mc mine':>7s}  blind")
for x in rows:
    print(f"{x['name']:18s} {x['catalog_snr']:7.1f} {x['near_snr']:6.1f} {x['near_snr'] / x['catalog_snr']:6.2f} "
          f"{x['near_dt_ms']:+6.1f}  {x['catalog_mc_det']:6.1f} {x['near_mc']:7.1f}  "
          f"{'EVENT' if x['blind_is_event'] else 'other'} {x['blind_snr']:.1f} at {x['blind_dt_s']:+.1f} s")
ok = [x for x in rows if x["blind_is_event"]]
print(f"\nloudest coincident trigger in 188 s of data is the catalog event for {len(ok)} of {len(rows)} events")
print(f"median SNR ratio (mine / catalog): {np.median([x['near_snr'] / x['catalog_snr'] for x in rows]):.3f}")
