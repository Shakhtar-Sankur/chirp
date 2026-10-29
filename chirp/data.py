"""Strain from the Gravitational Wave Open Science Center (GWOSC), and the event list it serves.

Everything is fetched over HTTPS from gwosc.org and cached on disk; nothing is redistributed.
GWOSC's certificate chain ends at a root that minimal systems may lack, so connections verify
against certifi's copy of the Mozilla CA list rather than skipping verification.
"""

import json
import os
import ssl
import urllib.request

import numpy as np

API = "https://gwosc.org/api/v2"


def _ctx():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def get_json(url):
    with urllib.request.urlopen(url, context=_ctx(), timeout=120) as r:
        return json.loads(r.read())


def download(url, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, url.rsplit("/", 1)[1])
    if not os.path.exists(path):
        tmp = path + ".part"
        with urllib.request.urlopen(url, context=_ctx(), timeout=600) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        os.replace(tmp, path)
    return path


def run_strain_urls(run, detector, start, stop, sample_rate_khz=4):
    """HDF5 URLs of the bulk run files (4096 s each) that cover [start, stop), in time order."""
    q = (f"{API}/runs/{run}/strain-files?detector={detector}&start={int(start)}&stop={int(stop)}"
         f"&sample-rate={sample_rate_khz}&format=json")
    rows, url = [], q
    while url:
        d = get_json(url)
        rows += [r for r in d["results"] if r["sample_rate_kHz"] == sample_rate_khz]
        url = d.get("next")
    rows.sort(key=lambda r: r["gps_start"])
    return [(r["gps_start"], r["hdf5_url"].split("?")[0]) for r in rows]


def strain_around(run, detector, gps, before, after, cache_dir, sample_rate_khz=4):
    """Strain from gps - before to gps + after (seconds), joined across 4096 s files.

    Returns (strain, gps_start, sample_rate); NaN where GWOSC has no science-quality data.
    """
    start, stop = gps - before, gps + after
    parts = []
    for _, url in run_strain_urls(run, detector, start, stop, sample_rate_khz):
        parts.append(read_hdf5(download(url, cache_dir)))
    if not parts:
        raise LookupError(f"no {run} {detector} data around GPS {gps}")
    fs = parts[0][2]
    t0 = parts[0][1]
    h = np.concatenate([p[0] for p in parts])
    i0 = int(round((start - t0) * fs))
    n = int(round((stop - start) * fs))
    if i0 < 0 or i0 + n > h.size:
        raise LookupError(f"{detector} files cover {t0}-{t0 + h.size / fs}, not {start}-{stop}")
    return h[i0:i0 + n], t0 + i0 / fs, fs


def read_hdf5(path):
    """(strain, gps_start, sample_rate). NaN marks samples outside GWOSC's science segments."""
    import h5py

    with h5py.File(path, "r") as f:
        h = f["strain/Strain"][()].astype(np.float64)
        dt = f["strain/Strain"].attrs["Xspacing"]
        t0 = f["meta/GPSstart"][()]
    return h, float(t0), 1.0 / dt
