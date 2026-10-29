# chirp

**An independent GPU search for gravitational waves in public LIGO data, with its own
waveform physics.**

LIGO, Virgo and KAGRA publish their strain data through the
[Gravitational Wave Open Science Center](https://gwosc.org). The O4a run (May 2023 – January 2024) was released
with the GWTC-4.0 catalog in August 2025. chirp searches that data from scratch and checks itself against
the catalog before it looks for anything new.

## What is here

- **`chirp/taylorf2.py`: the inspiral, from general relativity.** The stationary-phase TaylorF2 waveform to
  3.5 post-Newtonian order, written from the published closed-form coefficients (Blanchet 2014; Buonanno
  et al. 2009). Against LALSimulation's TaylorF2, its phase agrees to **6 × 10⁻¹⁰ rad** over 20 Hz to the
  innermost stable orbit, and its amplitude shape to machine precision, for every mass pair tested (neutron
  stars to 20 + 15 M☉; `tests/test_chirp.py`).
- **`chirp/search.py`: matched filtering.** Median-Welch noise spectra, inverse-spectrum truncation, and the
  SNR time series ρ(t) = |z(t)|/σ, the optimal detector for a known signal in Gaussian noise.
- **`chirp/bank.py`: a template bank placed by measured match.** Non-spinning binary black holes
  (IMRPhenomXAS), each step in chirp mass the largest that still matches the previous template at ≥ 0.97
  against a real O4a noise spectrum.
- **`chirp/gpu.py`: the search on a GPU.** Every event's data is whitened once; each template batch is
  generated once and filtered against all of them. The two LIGO detectors are combined into a coincident
  network SNR within the 10 ms light-travel time.
- **`chirp/veto.py`: telling signals from glitches.** Gating of loud transients in whitened strain, the
  Allen χ² discriminator over 16 frequency bins, and the reweighted SNR the PyCBC searches rank by. A
  second GPU pass rescores the loudest template in every 1 s of data (`notebooks/stage01_kaggle.py`).
- **`chirp/data.py`: GWOSC strain and the GWTC-4.0 event list**, fetched and cached, never redistributed.

## Results so far

**Waveform physics:** TaylorF2 matches LALSimulation to 6 × 10⁻¹⁰ rad in phase (above).

**The filter on the loudest O4a event, GW231226_101520** (catalog network SNR 34.7, detector-frame masses
49 + 43 M☉), with a small non-spinning grid around the published masses:

| | Hanford | Livingston | network |
|---|---|---|---|
| peak SNR | 26.07 | 21.14 | **33.56** (97% of the catalog's 34.7) |
| time of peak vs catalog | −1 ms | +3 ms | −1 ms |

Searching blind over the whole 188 s of valid data, the loudest coincident trigger is the event itself.
The remaining 3% is what non-spinning templates on a coarse grid cost against the best of the
collaboration's pipelines.

**Stage 0, every two-detector O4a event in the catalog.** `notebooks/stage0_kaggle.py` ran the full bank
(3886 templates) against all 76 GWTC-4.0 O4a events seen by both LIGO detectors with detector-frame chirp
mass 5–200 M☉. Five were skipped because one detector's data has gaps. The 71 that remain were searched on
one Kaggle Tesla T4 in 17 minutes ([public run](https://www.kaggle.com/code/sankurkundu/notebooke18924b056); two
runs gave identical results). Raw output and per-event numbers: `results/stage0_kaggle_2026-10-29/`.

| | result |
|---|---|
| network SNR near the catalog time, as a share of the catalog's | median **95%**; ≥ 90% for 61 of 71; lowest 78% (GW231005_091549) |
| time of the peak against the catalog | within 50 ms for 67 of 71, all within 98 ms |
| chirp mass against the catalog | within 10% for 51 of 71, and for 21 of the 22 below 30 M☉ |
| loudest coincident trigger in all 188 s is the event | **45 of 71** (48 within 0.1 s) |

What the misses show, and what stage 0.1 fixes:

- **Glitches.** In 7 of the 23 other cases the loudest trigger is an instrumental glitch, SNR 58 to 1824,
  far louder than any real signal. The search has no signal-consistency test yet. A χ² discriminator and
  gating of loud transients, as the collaboration's pipelines use, are the next step.
- **Noise near threshold.** In the other 16, a noise trigger at SNR 9 to 25 beats a quiet event (catalog
  SNR 8 to 12). The same χ² test separates these.
- **Heavy binaries.** For seven events with chirp mass above 50 M☉, the best template has a chirp mass
  30–60% low. These signals last only a few cycles in band, so a non-spinning bank trades mass against spin
  and mass ratio. Aligned-spin templates are the fix.

**Stage 0.1, the same search with gating and the χ² veto.** `notebooks/stage01_kaggle.py` gates glitches in
the whitened strain before estimating the spectrum, then rescores the loudest template in every 1 s of data
with the Allen χ² test over 16 frequency bins and the reweighted SNR the PyCBC searches rank by. One Kaggle
Tesla T4 ([public run](https://www.kaggle.com/code/sankurkundu/notebook43ba407303); raw output and per-event
numbers: `results/stage01_kaggle_2026-10-29/`).

| loudest trigger in 188 s is the catalog event (within 0.1 s) | events |
|---|---|
| stage 0, no veto | 48 of 71 |
| gating only | 49 of 71 |
| **gating and χ² reweighting** | **62 of 71** |

- The veto costs real signals little: at the events, the reweighted SNR is a median **98%** of the plain SNR.
  It falls most for the heaviest binaries (GW231123_135430: 19.1 → 12.9), where the non-spinning templates fit
  worst.
- Gating removed 56 glitches in the data around 32 of the events, and none within 1 s of an event.
- The 9 remaining misses are all quiet events (catalog SNR 13.4 or below). Noise triggers at reweighted SNR 7.6
  to 15.3 still edge them out. Telling those apart needs a measured background, which is stage 2.

## Plan

0. Re-find the published O4a events with a blind bank, and compare SNR, time and chirp mass with the catalog
   (done: above). 0.1: a χ² veto and gating (done: 62 of 71 blind); next, aligned-spin templates.
1. Measure sensitivity by injecting simulated signals into real O4a noise.
2. Estimate backgrounds by time slides and search below the catalog threshold, for signal classes the main
   pipelines cover less well.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests                                     # waveform, filter and bank checks
python scripts/fetch_events.py data/gwtc4_o4a_events.json      # event list from GWOSC
python scripts/make_bank.py /tmp/gwcache 5 200 data/bank_mc5_200.json
```

The Kaggle notebook (`notebooks/stage0_kaggle.py`) runs the search on a free T4 GPU.

## Credits

Data: LIGO Scientific Collaboration, Virgo Collaboration and KAGRA Collaboration, through GWOSC (O4a strain,
doi:10.7935/kt51-6n86; GWTC-4.0). IMR templates and the TaylorF2 reference: LALSuite (LIGO Scientific
Collaboration). This is an independent project, not a product of those collaborations.
