# GLP1R PGx RAW DATA — SOURCE BIBLIOGRAPHY
# Every variant in gnomAD_GLP1R_missense_variants.csv is traceable to these sources.

## PRIMARY GENOMIC SOURCES

1. gnomAD v4.1
   - 807,162 individuals (730,947 exomes + 76,215 genomes)
   - Includes 416,555 UK Biobank exomes
   - URL: https://gnomad.broadinstitute.org
   - All MAF values sourced from this release unless otherwise noted

2. UK Biobank
   - 200,000 exome-sequenced participants (subset used in Gao et al.)
   - Used for genotype-phenotype association (HbA1c, BMI)

## PRIMARY FUNCTIONAL CHARACTERIZATION SOURCES

3. Gao et al. (2023)
   - "Human GLP1R variants affecting GLP1R cell surface expression
     are associated with impaired glucose control and increased adiposity"
   - Nature Metabolism 5, 1673–1684
   - DOI: 10.1038/s42255-023-00889-6
   - DATA: 60 GLP1R variants profiled across 4 signaling pathways
     (cAMP, β-arrestin2, ERK1/2, Ca2+)
   - KEY FINDING: Three phenotypic clusters identified
     - Cluster 1 (red, 13 variants): Complete β-arr2 loss + drastically
       reduced cAMP potency + mid-high ERK efficacy loss
     - Cluster 2 (blue, 7 variants): Detectable β-arr2 with drastic
       Emax/Ka losses + reduced ERK efficacy + increased ERK potency
     - Cluster 3 (black, 36 variants): Variable impairments
   - KEY FINDING: A316T is GoF for cAMP/Ca2+, LoF for ERK/β-arr2
     (biased receptor)
   - KEY FINDING: Impaired cell surface expression → poor glucose
     control + increased adiposity (UK Biobank association)

4. Lagou et al. (2023)
   - "GWAS of random glucose in 476,326 individuals provide insights
     into diabetes pathophysiology, complications and treatment stratification"
   - Nature Genetics 55, 1448–1461
   - DOI: 10.1038/s41588-023-01462-3
   - DATA: 17 GLP1R coding variants functionally tested
   - KEY FINDING: A316T (rs10305492) is the lead missense variant with
     strong glucose-lowering effect (0.058 mmol/L per allele)
   - KEY FINDING: MD simulations of G168S showed increased flexibility
     at the 168-178 region (increased distance 1.63-Y178 2.48)

5. Nature 2026 (April)
   - "Genetic predictors of GLP1 receptor agonist weight loss and
     side effects"
   - Nature (2026)
   - DOI: 10.1038/s41586-026-10330-z
   - DATA: GWAS of 27,885 GLP1-RA treated individuals
   - KEY FINDING: rs10305420 (P7L) associated with increased efficacy
     (P = 2.9×10⁻¹⁰, −0.76 kg per effect allele)
   - KEY FINDING: 7 GLP1R missense variants with MAF >1% in gnomAD v4.1:
     rs1042044, rs10305420, rs3765467, rs10305421, rs2295006,
     rs10305510, rs201672448

6. Danish Cohort Study
   - "Rare Heterozygous Loss-of-Function Variants in the Human GLP-1
     Receptor Are Not Associated With Cardiometabolic Phenotypes"
   - PMC10584003
   - DATA: 36 nonsynonymous GLP1R variants from 8,642 Danish individuals
     (2,930 T2D patients + 5,712 population-based)
   - KEY FINDING: 10 variants with complete loss of cAMP signaling (LoS)
     - EC50 >370 pmol/L or Emax <50% WT
   - KEY FINDING: 26 variants WT-like (<5-fold potency shift)

7. Hinds et al. (2024)
   - "Abolishing β-arrestin recruitment is necessary for the full
     metabolic benefits of G protein-biased GLP-1 receptor agonists"
   - Diabetes, Obesity and Metabolism
   - DOI: 10.1111/dom.15288
   - DATA: gnomAD coding variants tested with biased agonist SRB107
   - KEY FINDING: D344E (rs2295006) differentially modulated by biased
     vs unbiased agonists — pharmacogenomic implications

8. Bitsi et al. (2026)
   - "In vivo functional profiling and structural characterisation of
     the human GLP1R A316T variant"
   - Science Advances, eadw0899
   - DOI: 10.1126/sciadv.adw0899
   - DATA: Cryo-EM structure of A316T GLP-1R bound to GLP-1 + MD sims +
     human GLP1R A316T knock-in mouse model
   - KEY FINDING: A316T shows constitutive activation (lower fasting
     glucose) BUT blunted pharmacological GLP-1RA response in vivo
   - KEY FINDING: Altered TM5-TM6 interface geometry

9. Li et al. (2020)
   - "GLP1R Single-Nucleotide Polymorphisms rs3765467 and rs10305492
     Affect β Cell Insulin Secretory Capacity and Apoptosis Through GLP-1"
   - DNA and Cell Biology 39(9), 1700-1710
   - DOI: 10.1089/dna.2020.5424
   - DATA: In vitro β-cell assays for rs3765467 and rs10305492
   - KEY FINDING: Both SNPs reduce insulin secretion and cAMP, promote
     β-cell apoptosis under high glucose

10. Beinborn et al. (2005)
    - Original characterization of T149M reducing GLP-1 binding affinity
    - Referenced in Gao et al. 2023 and gnomAD annotations

## STRUCTURAL SOURCES

11. PDB 6XOX
    - GLP-1R active state cryo-EM structure, Gs-coupled
    - Resolution: 3.3 Å
    - Used for: domain boundaries, residue position mapping,
      allosteric pathway identification

12. PDB 5VEX
    - GLP-1R inactive state, NAM-bound
    - Used for: inactive state comparison, cold-dominant validation

13. GPCRdb
    - Ballesteros-Weinstein numbering for Class B1 GPCRs
    - URL: https://gpcrdb.org/protein/glp1r_human/
    - Used for: cross-receptor residue alignment, conserved motif
      identification (NPxxY, DRY equivalent)
