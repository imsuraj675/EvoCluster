#!/usr/bin/env python3
import os
import numpy as np
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from homology_corr import hm_correlation_analysis

# ===========================
# Configuration
# ===========================
pfam_ids = [
    "PF04961", "PF01668", "PF05635",
    "PF00158", "PF12002", "PF01948",
    "PF01196", "PF20415", "PF03914",
    "PF01156", "PF11799", "PF13356"
]

models = ["esmc", "esm2", "esm1", "pt", "msa"]
shuffle = "N"
colattn = "N"

# Output directory
os.makedirs("./results", exist_ok=True)

# ===========================
# Thread-safe counters
# ===========================
lock = threading.Lock()
passed = failed = 0
total = len(pfam_ids) * len(models)

# ===========================
# Worker function for a single model
# ===========================
def run_model(pfam_id, model):
    seed_file = f"{pfam_id}/{pfam_id}.aln"
    try:
        ESS_spearman, ESS_pearson = hm_correlation_analysis(seed_file, model, shuffle, colattn)
        np.save(f"results/{model}_{pfam_id}_spear.npy", np.array(ESS_spearman))
        np.save(f"results/{model}_{pfam_id}_pear.npy", np.array(ESS_pearson))
        return (pfam_id, model, True)
    except Exception as e:
        print(f"❌ {model.upper()} - {pfam_id} failed: {e}")
        return (pfam_id, model, False)

# ===========================
# Process a single Pfam: prep + run models concurrently
# ===========================
def process_pfam(pfam_id):
    global passed, failed
    seed_file = f"{pfam_id}/{pfam_id}.aln"
    if not os.path.exists(seed_file):
        print(f"⚠️ Missing alignment file for {pfam_id}, skipping.")
        with lock:
            failed += len(models)
        return

    # Run prep_data.py once
    try:
        subprocess.run(["python", "../src/prep_data.py", "-a", seed_file], check=True)
        print(f"✅ Prep done for {pfam_id}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Prep failed for {pfam_id}: {e}")
        with lock:
            failed += len(models)
        return

    # Run all models concurrently for this Pfam
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = [executor.submit(run_model, pfam_id, model) for model in models]
        for future in as_completed(futures):
            pfam_id_res, model_res, success = future.result()
            with lock:
                if success:
                    passed += 1
                    status = "✅"
                else:
                    failed += 1
                    status = "❌"
                remaining = total - (passed + failed)
                print(f"{status} {model_res.upper()} - {pfam_id_res} | Passed {passed} Failed {failed} Remaining {remaining}")

# ===========================
# Main execution: all Pfams sequentially
# ===========================
for pfam in pfam_ids:
    process_pfam(pfam)

print("\n🎯 All analyses complete!")
print(f"Summary → Passed {passed} | Failed {failed} | Total {total}")
print("Results saved in ./results/")
# ===========================