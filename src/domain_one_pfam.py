import os
import numpy as np
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils_get_embeddings import get_plm_representation
from homology_corr import layer_wise_lg_corr


# --------------------------------------------------------------------
# Process ONE PFAM for ONE model
# --------------------------------------------------------------------
def process_pfam_model(pfam, model_type):
    try:
        aln_file = f"../data/{pfam}/{pfam}_domain.aln"
        lg_file = f"../data/{pfam}/lg_mat.npy"
        seq_ref_file = f"../data/{pfam}/sequence_ref_dict.pkl"
        if os.path.exists(f"results_domain/{model_type}_{pfam}_domain_spear.npy"):
            return (pfam, model_type, False, f"[SKIP] Skipped {aln_file} {model_type}")

        # Required files
        if not os.path.exists(aln_file):
            return (pfam, model_type, False, f"[WARN] Missing ALN: {aln_file}")

        if not os.path.exists(lg_file):
            return (pfam, model_type, False, f"[WARN] Missing LG: {lg_file}")

        if not os.path.exists(seq_ref_file):
            return (pfam, model_type, False, f"[WARN] Missing seq_ref_idx.pkl: {seq_ref_file}")

        # Load reference index
        seq_ref = pickle.load(open(seq_ref_file, "rb"))

        # Load LG matrix
        lg_mat = np.load(lg_file)

        # Load representation
        plmdata = get_plm_representation(
            model_type,
            aln_file,
            aln_file,
            "all",
            True
        )

        # Correlation
        rho_dict, pearson_dict = layer_wise_lg_corr(plmdata, lg_mat, seq_ref)

        # Save outputs
        os.makedirs("results_domain", exist_ok=True)

        rho_out = f"results_domain/{model_type}_{pfam}_domain_spear.npy"
        pearson_out = f"results_domain/{model_type}_{pfam}_domain_pearson.npy"

        np.save(rho_out, rho_dict)
        np.save(pearson_out, pearson_dict)

        return (pfam, model_type, True, None)

    except Exception as e:
        return (pfam, model_type, False, str(e))


# --------------------------------------------------------------------
# Wrapper so each worker processes ALL PFAMs for ONE model
# --------------------------------------------------------------------
def process_model(model_type, pfam_list):
    results = []
    for pfam in pfam_list:
        results.append(process_pfam_model(pfam, model_type))
    return results


# --------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------
def main():

    MODEL_LIST = ["esm2", "esmc", "pt", "esm1"]  # workers = len(list)

    # Load PFAM list
    with open("all_pfams.txt", "r") as f:
        pfams = [line.strip() for line in f if line.strip()]

    print(f"Found {len(pfams)} PFAMs.")

    total_jobs = len(MODEL_LIST) * len(pfams)
    passed = 0
    failed = 0

    # Run len(models) workers → each worker handles one model
    with ThreadPoolExecutor(max_workers=len(MODEL_LIST)) as executor:
        futures = {
            executor.submit(process_model, model, pfams): model
            for model in MODEL_LIST
        }

        with tqdm(total=total_jobs, desc="Running jobs") as pbar:
            for future in as_completed(futures):
                model_type = futures[future]

                try:
                    model_results = future.result()
                    for pfam, model, success, msg in model_results:
                        if success:
                            passed += 1
                        else:
                            failed += 1
                            print(f"❌ {model}-{pfam} failed: {msg}")
                        pbar.update(1)

                except Exception as e:
                    print(f"[FATAL] Worker for model {model_type} crashed: {e}")

    print("\n🎯 DONE")
    print(f"Passed = {passed}")
    print(f"Failed = {failed}")
    print(f"Total  = {total_jobs}")


if __name__ == "__main__":
    main()

