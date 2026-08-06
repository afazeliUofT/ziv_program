#!/usr/bin/env python3
"""v4 verification suite: A1 corrected-lemma census, A2 converse
constants, A3 completed L2 census through n=16.
Usage: python run_v4.py [--scale smoke|full]   Full ~5-15 min."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from src_v4 import experiments_v4 as X

ap = argparse.ArgumentParser()
ap.add_argument("--scale", choices=["smoke", "full"], default="full")
ap.add_argument("--seed", type=int, default=20260823)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
t0 = time.time(); man = {"scale": a.scale, "seed": a.seed, "timings_s": {}}
print("[v4] frontier ...", flush=True); t=time.time()
frontier = X.build_frontier(); man["timings_s"]["frontier"]=round(time.time()-t,1)
res = {}
for name, fn in [("a1", lambda: X.a1_lemma_census(rng, a.scale)),
                 ("a2", lambda: X.a2_converse_constants(a.scale)),
                 ("a3", lambda: X.a3_census_completion(frontier, a.scale))]:
    print(f"[v4] {name} ...", flush=True); t=time.time()
    res[name]=fn(); man["timings_s"][name]=round(time.time()-t,1)
    print(f"[v4]   done ({man['timings_s'][name]}s)", flush=True)
card = {"A1": {"verdict": res["a1"]["verdict"],
               "sequences_checked": res["a1"]["sequences_checked"],
               "corrected_violations": res["a1"]["corrected_violations"],
               "retracted_variant_violations":
                   res["a1"]["retracted_variant_violations"]},
        "A2_rows": res["a2"]["rows"],
        "A3": {k: res["a3"][k] for k in ("total_types","n_failures",
              "coverage","max_radius","n_range","certified")},
        "v41_gate": ("FINAL_CENSUS_CERTIFIED_RADIUS_2S"
                    if res["a1"]["verdict"]=="CORRECTED_LEMMA_VERIFIED"
                    and res["a3"]["certified"] else "REVIEW_REQUIRED")}
os.makedirs("results_v4", exist_ok=True)
json.dump(card, open("results_v4/v4_scorecard.json","w"), indent=1,
          default=float)
man["total_s"]=round(time.time()-t0,1); man["v41_gate"]=card["v41_gate"]
json.dump(man, open("V4_MANIFEST.json","w"), indent=1)
dig = {}
if os.path.exists("DIGEST.json"):
    try: dig=json.load(open("DIGEST.json"))
    except Exception: dig={}
dig["V4_MANIFEST"]=man
for fn in sorted(os.listdir("results_v4")):
    if fn.endswith(".json"):
        try: dig["v4/"+fn]=json.load(open(os.path.join("results_v4",fn)))
        except Exception as e: dig["v4/"+fn]={"error":str(e)}
json.dump(dig, open("DIGEST.json","w"), indent=1, default=float)
print("\n=== V4 SCORECARD ===")
print(json.dumps({k:card[k] for k in ("A1","A3","v41_gate")}, indent=1,
                 default=float))
print("Details: results_v4/ ; merged into DIGEST.json")
