import os
from concurrent.futures import ProcessPoolExecutor, as_completed

ALL_PFAMS = "all_pfams.txt"

def run_pfam(pfam):
    pfam = pfam.strip()
    if not pfam:
        return f"{pfam}: skipped (empty)"

    aln_file = f"../data/{pfam}/{pfam}.aln"

    if not os.path.exists(aln_file):
        return f"{pfam}: ❌ missing {aln_file}"

    # run silently (stdout and stderr -> /dev/null)
    cmd = f"python prep_data.py -a {aln_file} > /dev/null 2>&1"
    exit_code = os.system(cmd)

    if exit_code == 0:
        return f"{pfam}: ✔ done"
    else:
        return f"{pfam}: ❌ failed ({exit_code})"

def main():
    with open(ALL_PFAMS) as f:
        pfams = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(pfams)} PFAMs.")
    print("Running with 8 parallel jobs silently...\n")

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_pfam, pfam): pfam for pfam in pfams}

        for future in as_completed(futures):
            pfam = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"{pfam}: ❌ crashed with exception: {e}")
            else:
                print(result)

    print("\nAll PFAM jobs finished.")

if __name__ == "__main__":
    main()

