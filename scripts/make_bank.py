"""Write a template bank placed by match against a measured O4a noise spectrum.

    python scripts/make_bank.py CACHE_DIR MC_MIN MC_MAX OUT.json

The spectrum is Livingston's, from 800 s around GW231226_101520; templates are IMRPhenomXAS,
non-spinning, f_low = 20 Hz, matched over 64 s segments at 4096 Hz.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chirp import bank, data, search  # noqa: E402

FS, T, F_LOW, MIN_MATCH = 4096.0, 64.0, 20.0, 0.97

if __name__ == "__main__":
    cache, mc_min, mc_max, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    f = np.fft.rfftfreq(int(T * FS), 1 / FS)
    h, _, _ = data.strain_around("O4a", "L1", 1387620938.3, 600, 200, cache)
    pf, psd = search.welch_psd(h, FS)
    inv = search.inverse_psd(pf, psd, f, FS, F_LOW)
    t = time.time()
    b = bank.build(f, inv, F_LOW, mc_min=mc_min, mc_max=mc_max, min_match=MIN_MATCH)
    print(len(b), "templates", f"{time.time() - t:.0f} s")
    json.dump({"approximant": "IMRPhenomXAS", "f_low": F_LOW, "min_match": MIN_MATCH, "mass_ratios": list(bank.QS),
               "psd": "L1, 800 s around GW231226_101520", "columns": ["m1", "m2", "mc", "q"], "templates": b},
              open(out, "w"))
