#!/usr/bin/env bash

# Check input
if [ $# -lt 2 ]; then
    echo "Usage: $0 <PfamID> <seed|full>"
    exit 1
fi

pfam_id="$1"
mode="$2"

echo "Pfam number: $pfam_id"
echo "Alignment type: $mode"

# Set directory based on mode
if [ "$mode" == "seed" ]; then
    outdir="${pfam_id}"
    urlsuffix="?annotation=alignment:seed&download"
elif [ "$mode" == "full" ]; then
    outdir="${pfam_id}_f"
    urlsuffix="?annotation=alignment:full&download"
else
    echo "Invalid alignment type. Use 'seed' or 'full'."
    exit 1
fi

# Create output directory
mkdir -p "$outdir"

tmpfile="${outdir}/${outdir}.sto.gz"
outfile="${outdir}/${outdir}.aln"
urlprefix="https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam"
url="${urlprefix}/${pfam_id}/${urlsuffix}"

echo "Downloading from: $url"
wget -O "$tmpfile" "$url"

if [ ! -f "$tmpfile" ]; then
    echo "Download failed."
    exit 1
fi

echo "Extracting, cleaning, and formatting alignment..."
zcat "$tmpfile" | grep -Ev "^#" | grep -Ev "^//" | \
awk '{print ">" $1 "\n" $2}' > "$outfile"

echo "Saved formatted alignment to: $outfile"
