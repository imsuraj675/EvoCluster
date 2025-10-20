#!/usr/bin/env bash

# 12 diverse Pfam IDs for homology benchmarking
pfam_ids=(
    PF04961 PF01668 PF05635
    PF00158 PF12002 PF01948
    PF01196 PF20415 PF03914
    PF01156 PF11799 PF13356
)

urlprefix="https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam"

for pfam_id in "${pfam_ids[@]}"; do
    echo "=========================================="
    echo "Processing Pfam: $pfam_id (seed alignment)"
    echo "------------------------------------------"

    outdir="$pfam_id"
    outfile="${outdir}/${pfam_id}.aln"

    # Skip if already exists
    if [ -s "$outfile" ]; then
        echo "✅ $outfile already exists. Skipping."
        continue
    fi

    mkdir -p "$outdir"

    tmpfile="${outdir}/${pfam_id}.sto.gz"
    url="${urlprefix}/${pfam_id}/?annotation=alignment:seed&download"

    echo "Fetching from: $url"
    wget -q -O "$tmpfile" "$url"

    if [ ! -s "$tmpfile" ]; then
        echo "❌ Download failed for $pfam_id"
        continue
    fi

    echo "✅ Download complete, converting to FASTA..."
    zcat "$tmpfile" | grep -Ev "^#" | grep -Ev "^//" | \
    awk '{print ">" $1 "\n" $2}' > "$outfile"

    echo "✅ Saved FASTA alignment to: $outfile"
done

echo "=========================================="
echo "🎯 All 12 Pfam seed alignments downloaded successfully!"
