"""Write data/gwtc4_o4a_events.json: every GWTC-4.0 event with its search SNR and masses.

    python scripts/fetch_events.py data/gwtc4_o4a_events.json

Search SNR is the largest network matched-filter SNR over the search pipelines; masses come from
the preferred parameter-estimation set. GWTC-4.0 gives source-frame masses and redshift, so the
detector-frame chirp mass (what a template bank sees) is Mc_source * (1 + z).
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chirp.data import API, get_json  # noqa: E402

KEEP = ("chirp_mass_source", "mass_1_source", "mass_2_source", "redshift", "chi_eff")


def events():
    url, out = f"{API}/catalogs/GWTC-4.0/events?format=json&pagesize=100", []
    while url:
        d = get_json(url)
        out += d["results"]
        url = d["next"]
    return out


def one(e):
    pv = get_json(e["detail_url"].split("?")[0] + "/parameters?format=json")["results"]
    search = [p for p in pv if p["pipeline_type"] == "search"]
    snr = [q["best"] for p in search for q in p["parameters"] if q["name"] == "network_matched_filter_snr" and q["best"]]
    far = [q["best"] for p in search for q in p["parameters"] if q["name"] == "far" and q["best"] is not None]
    pe = sorted((p for p in pv if p["pipeline_type"] != "search"), key=lambda p: not p["is_preferred"])
    vals = {q["name"]: q["best"] for q in (pe[0]["parameters"] if pe else [])}
    mc, z = vals.get("chirp_mass_source"), vals.get("redshift")
    return {"name": e["name"], "gps": e["gps"], "detectors": e["detectors"],
            "snr_search_max": max(snr) if snr else None, "far_min": min(far) if far else None,
            "chirp_mass_det": mc * (1 + z) if mc and z is not None else None, **{k: vals.get(k) for k in KEEP}}


if __name__ == "__main__":
    with ThreadPoolExecutor(8) as ex:
        rows = sorted(ex.map(one, events()), key=lambda r: r["gps"])
    json.dump({"source": "GWOSC API v2, catalog GWTC-4.0 (O4a); search SNR = max over search pipelines, "
                         "masses from the preferred PE set", "events": rows}, open(sys.argv[1], "w"), indent=1)
    print(len(rows), "events")
