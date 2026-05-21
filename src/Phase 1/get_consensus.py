
#!/usr/bin/env python3
import os
import subprocess
from collections import Counter
from Bio import AlignIO, SeqIO
from Bio.Align import AlignInfo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# === Configuration ===
pfam_ids = [
    "PF04298", "PF07907", "PF02702", "PF04961", "PF04461", "PF04403", "PF01668", "PF01450", "PF04304", "PF03975",
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

consensus_dir = "../data/all_consensus"
os.makedirs(consensus_dir, exist_ok=True)

combined_fa = os.path.join(consensus_dir, "all_consensus.fasta")
combined_aln = os.path.join(consensus_dir, "all_consensus.aln")

# === Step 1: Generate consensus sequences ===

def consensus_from_fasta(fasta_file, ignore_gaps=False):
    """
    Compute consensus sequence from an MSA FASTA file by taking
    the most frequent character in each column.

    Parameters
    ----------
    fasta_file : str
        Path to the input FASTA alignment file.
    ignore_gaps : bool, optional
        If True, gaps ('-') are ignored when finding the most frequent residue.

    Returns
    -------
    str
        Consensus sequence string.
    """
    # Read all sequences
    records = list(SeqIO.parse(fasta_file, "fasta"))
    if not records:
        raise ValueError("No sequences found in the FASTA file.")

    seq_length = len(records[0].seq)
    for rec in records:
        if len(rec.seq) != seq_length:
            raise ValueError("All sequences must be of the same length.")

    # Build consensus
    consensus = []
    for i in range(seq_length):
        column = [rec.seq[i] for rec in records]
        if ignore_gaps:
            column = [res for res in column if res != '-']
        if not column:  # all gaps
            consensus.append('-')
        else:
            most_common = Counter(column).most_common(1)[0][0]
            consensus.append(most_common)

    return ''.join(consensus)

records = []

for pfam in pfam_ids:
    aln_path = f"../data/{pfam}/{pfam}.aln"
    if not os.path.exists(aln_path):
        print(f"⚠️ Missing: {aln_path}")
        continue

    try:
        '''
        # Try reading as FASTA first, fall back to Stockholm
        try:
            alignment = AlignIO.read(aln_path, "fasta")
        except Exception:
            alignment = AlignIO.read(aln_path, "stockholm")

        summary = AlignInfo.SummaryInfo(alignment)
        '''
        consensus_seq = consensus_from_fasta(aln_path, True) # summary.dumb_consensus(threshold=0.7, ambiguous="X")
        
        rec = SeqRecord(
            Seq(str(consensus_seq).replace("?", "-")),
            id=pfam,
            description="Consensus sequence"
        )
        records.append(rec)
        print(f"✅ Consensus created for {pfam}")

    except Exception as e:
        print(f"❌ Failed {pfam}: {e}")

# Write all to single FASTA file
SeqIO.write(records, combined_fa, "fasta")
print(f"\n🧬 Combined consensus written to: {combined_fa}")


# === Step 2: Align combined consensus using Clustal Omega ===
cmd = [
    "clustalo",
    "-i", combined_fa,
    "-o", combined_aln,
    "--outfmt=fasta",
    "--force"
]
try:
    subprocess.run(cmd, check=True)
    print(f"✅ Alignment complete → {combined_aln}")
except subprocess.CalledProcessError as e:
    print(f"❌ Clustal Omega failed: {e}")
