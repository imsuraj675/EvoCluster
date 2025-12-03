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
pfam_ids=["PF04298", "PF07907", "PF02702", "PF04961", "PF04461", "PF04403", "PF01668", "PF01450", "PF04304", "PF03975",
"PF05635", "PF19609", "PF03862", "PF01219", "PF06421", "PF14842", "PF03788", "PF00934", "PF02049", "PF02617",
"PF04341", "PF01052", "PF02700", "PF14841", "PF00986", "PF04839", "PF02805", "PF01313", "PF00831", "PF02073",
"PF00158", "PF09278", "PF12002", "PF12775", "PF17491", "PF16658", "PF01706", "PF02261", "PF14278", "PF09361",
"PF18565", "PF17146", "PF01948", "PF00707", "PF06071", "PF10437", "PF13667", "PF00189", "PF11760", "PF02650",
"PF10369", "PF01037", "PF06130", "PF02594", "PF07554", "PF09269", "PF00366", "PF03719", "PF03880", "PF20554",
"PF01055", "PF12686", "PF20060", "PF03914", "PF02586", "PF02365", "PF01139", "PF00617", "PF03150", "PF01297", 
"PF20415", "PF00856", "PF00902", "PF02104", "PF02811", "PF12276", "PF01368", "PF02383", "PF04051", "PF12704", 
"PF02517", "PF01728", "PF01885", "PF00636", "PF01196", "PF13423", "PF01926", "PF04205", "PF04427", "PF13508", 
"PF04264", "PF05192", "PF01156", "PF11799", "PF03796", "PF01288", "PF09994", "PF13280", "PF16124", "PF13356", 
"PF17852", "PF12804", "PF01510", "PF04122", "PF01369", "PF01302", "PF01266", "PF00006", "PF12697", "PF00557", 
"PF01388", "PF05257", "PF05226", "PF01149"
]

models = ["esmc", "esm2", "pt", "esm1"]
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
    # Check if result files already exist
    global passed,failed
    spear_file = f"results/{model}_{pfam_id}_spear.npy"
    pear_file = f"results/{model}_{pfam_id}_pear.npy"
    if model == "msa":
        return (pfam_id, model, False)
    if os.path.exists(spear_file) and os.path.exists(pear_file):
        return (pfam_id, model, True)
    seed_file = f"../data/{pfam_id}/{pfam_id}.aln"
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

    seed_file = f"../data/{pfam_id}/{pfam_id}.aln"
    if not os.path.exists(seed_file):
        print(f"  Missing alignment file for {pfam_id}, skipping.")
        with lock:
            failed += len(models)
        return


    # Run prep_data.py once
    #try:
     #   subprocess.run(["python", "../src/prep_data.py", "-a", seed_file], check=True)
      #  print(f"✅ Prep done for {pfam_id}")
    #except subprocess.CalledProcessError as e:
     #   print(f"❌ Prep failed for {pfam_id}: {e}")
      #  with lock:
       #     failed += len(models)
        #return


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
