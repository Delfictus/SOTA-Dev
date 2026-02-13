#!/bin/bash
mkdir -p e2e_validation_test/fpocket_baseline

# Loop through all topology files to find which PDBs we need
for topo in e2e_validation_test/prep/*.topology.json; do
    # Extract PDB ID (e.g., "1a4q" from "1a4q.topology.json")
    filename=$(basename "$topo")
    pdb_id="${filename%%.*}"
    
    # Define paths
    pdb_file="e2e_validation_test/prep/${pdb_id}.pdb"
    out_dir="e2e_validation_test/fpocket_baseline/${pdb_id}_out"
    
    # Skip if Fpocket data already exists
    if [ -d "$out_dir" ]; then
        echo "✅ Found data for $pdb_id"
        continue
    fi

    # Download PDB if missing
    if [ ! -f "$pdb_file" ]; then
        echo "⬇️  Downloading $pdb_id..."
        wget -q "https://files.rcsb.org/download/${pdb_id^^}.pdb" -O "$pdb_file"
        
        # Check if download succeeded
        if [ ! -s "$pdb_file" ]; then
            echo "❌ Failed to download $pdb_id"
            rm "$pdb_file"
            continue
        fi
    fi

    # Run Fpocket
    echo "⚙️  Running Fpocket on $pdb_id..."
    fpocket -f "$pdb_file" > /dev/null 2>&1
    
    # Move output to baseline folder
    # Fpocket creates output in the same dir as input
    if [ -d "e2e_validation_test/prep/${pdb_id}_out" ]; then
        mv "e2e_validation_test/prep/${pdb_id}_out" "$out_dir"
    fi
done
