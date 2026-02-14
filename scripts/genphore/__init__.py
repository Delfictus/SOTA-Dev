"""WT-1: Spike-to-pharmacophore conversion and generative molecule design.

Converts PRISM spike JSON → SpikePharmacophore, then runs PhoreGen (diffusion)
and PGMG (VAE) to produce List[GeneratedMolecule].
"""
