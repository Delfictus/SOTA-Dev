flatpak run --filesystem=/home/diddy/Desktop org.pymol.PyMOL \
  /home/diddy/Desktop/PRISM4D-bio/data/raw/2VWD.pdb \
  /home/diddy/Desktop/PRISM4D-bio/data/processed/nipah_relaxed_perfect.pdb











# 1. Setup the View
hide all
bg_color white
show cartoon
color green, 2VWD
color red, nipah_relaxed_perfect

# 2. Align the "Frozen" Core to ensure accurate measurement
align nipah_relaxed_perfect, 2VWD

# 3. Run the Python Analysis Engine
python
print("\n" + "="*60)
print("🧬 PRISM-ZERO: TOP 10 MOBILE RESIDUES (SURGICAL SCAN)")
print("="*60)
print(f"{'RANK':<5} | {'RESIDUE':<15} | {'DIST (Å)':<10}")
print("-" * 60)

# Get all Alpha Carbons from the reference structure
atoms = []
cmd.iterate("2VWD and name CA", "atoms.append((chain, resi, resn))")

results = []

for chain, resi, resn in atoms:
    # Define the two corresponding atoms
    sel1 = f"/2VWD//{chain}/{resi}/CA"
    sel2 = f"/nipah_relaxed_perfect//{chain}/{resi}/CA"
    
    try:
        # Calculate Euclidean distance
        dist = cmd.get_distance(sel1, sel2)
        results.append((dist, chain, resi, resn))
    except:
        pass

# Sort by distance (Highest first)
results.sort(key=lambda x: x[0], reverse=True)

# Print Top 10 Report
for i in range(10):
    dist, chain, resi, resn = results[i]
    label = f"{resn} {chain}:{resi}"
    print(f"{i+1:<5} | {label:<15} | {dist:<10.2f}")
    
    # Highlight the #1 Winner visually
    if i == 0:
        # Create a selection for the top hit
        cmd.select("top_hit", f"/nipah_relaxed_perfect//{chain}/{resi}/CA")
        # Show it as a Yellow Sphere
        cmd.show("spheres", "top_hit")
        cmd.color("yellow", "top_hit")
        # Draw the yellow dotted line showing the movement
        cmd.distance("max_displacement", f"/2VWD//{chain}/{resi}/CA", f"/nipah_relaxed_perfect//{chain}/{resi}/CA")
        # Label it
        cmd.set("label_size", 20)
        cmd.set("label_color", "black")

print("="*60 + "\n")
python end

# 4. Zoom to the winner
zoom top_hit, 20
