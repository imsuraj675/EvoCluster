from ete3 import Tree
import os

def merge_pfam_trees(pfam_list,
                     data_dir="../data",
                     consensus_tree_path="../data/all_consensus/all_consensus.tree",
                     output_tree="../data/all_consensus/all_pfams_merged.tree"):
    """
    Merge all individual Pfam LG trees under the existing consensus tree structure.
    Each Pfam node in the consensus tree gets its corresponding LG tree attached.
    """

    # Load the existing consensus backbone tree
    if not os.path.exists(consensus_tree_path):
        raise FileNotFoundError(f"Consensus tree not found at {consensus_tree_path}")

    master_tree = Tree(consensus_tree_path, format=1)
    print(f"Loaded consensus tree: {consensus_tree_path}")

    # Attach each Pfam-specific subtree
    for pfam in pfam_list:
        pfam_tree_path = os.path.join(data_dir, pfam, f"{pfam}.tree")
        if not os.path.exists(pfam_tree_path):
            print(f"⚠️ Skipping {pfam}: no tree file found.")
            continue

        try:
            sub_tree = Tree(pfam_tree_path, format=1)
        except Exception as e:
            print(f"❌ Failed to read {pfam}.tree: {e}")
            continue

        # Find the node in the consensus tree corresponding to this Pfam
        node = master_tree.search_nodes(name=pfam)
        if not node:
            print(f"⚠️ Pfam {pfam} not found in consensus tree; adding at root.")
            node = master_tree.add_child(name=pfam)
        else:
            node = node[0]

        # Attach this Pfam's LG tree as a child under its node
        node.add_child(sub_tree)

    # Save the final merged tree
    master_tree.write(format=1, outfile=output_tree)
    Tree(output_tree, format = 1)
    print(f"✅ Merged tree written to {output_tree}")


if __name__ == "__main__":
    pfam_list =[
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
    merge_pfam_trees(pfam_list)

