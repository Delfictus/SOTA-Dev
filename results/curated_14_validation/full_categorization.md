# Curated 14 Structures - Full Categorization

Generated: 2026-01-17

## MASTER TABLE

| PDB | Family | Tier | Size | Atoms | OpenMM PE | Engine PE | Status | Domain Use |
|-----|--------|------|------|-------|-----------|-----------|--------|------------|
| 1L2Y | Synthetic | Benchmark | Tiny | 304 | -510 | 8.14e+04 | ✅ READY | Folding benchmark |
| 4QWO | Lassa | Emerging | Small | 4,148 | -7,155 | 1.18e+06 | ✅ READY | Vaccine target |
| 6LU7 | SARS-CoV-2 | Pandemic | Medium | 4,730 | -8,182 | 1.38e+06 | ✅ READY | Drug target (Paxlovid) |
| 1AKE | Human | Druggable | Medium | 6,682 | -12,682 | 1.76e+06 | ✅ READY | Cryptic site benchmark |
| 3SQQ | Marburg | Emerging | Medium | 4,705 | -7,533 | 8.10e+10 | ⚠️ ELEVATED | Immune evasion |
| 1HXY | HIV-1 | Pandemic | Large | 9,444 | -15,594 | 3.24e+10 | ⚠️ ELEVATED | Vaccine target |
| 3HEC | Bacterial | Druggable | Medium | 5,351 | -9,109 | 4.83e+11 | ⚠️ ELEVATED | Enzyme target |
| 4Z0J | Norovirus | Emerging | Medium | 7,798 | -14,786 | 6.52e+12 | ❌ HIGH | Capsid target |
| 2F4J | Human | Druggable | Small | 4,608 | -6,508 | 7.78e+14 | ❌ HIGH | Kinase target |
| 6M0J | SARS-CoV-2 | Pandemic | Large | 12,510 | -18,695 | 6.84e+13 | ❌ HIGH | Vaccine/escape |
| 2VWD | Nipah | Pandemic | Large | 12,926 | -22,914 | 2.86e+14 | ❌ HIGH | Therapeutic target |
| 4J1G | Influenza | Pandemic | XL | 15,799 | -27,193 | 2.15e+14 | ❌ HIGH | HA target |
| 4B7Q | Influenza | Pandemic | XL | 23,312 | -41,907 | 1.25e+15 | ❌ HIGH | NA resistance |
| 5IRE | Zika | Pandemic | XL | 26,297 | -38,952 | 1.70e+14 | ❌ HIGH | Vaccine target |

---

## BY STATUS

### ✅ Production Ready (4 structures)
| PDB | Family | Atoms | Engine PE | Use Case |
|-----|--------|-------|-----------|----------|
| 1L2Y | Synthetic | 304 | 8.14e+04 | Protein folding benchmark |
| 4QWO | Lassa | 4,148 | 1.18e+06 | Emerging virus vaccine |
| 6LU7 | SARS-CoV-2 | 4,730 | 1.38e+06 | Antiviral drug target |
| 1AKE | Human | 6,682 | 1.76e+06 | Apo-holo cryptic site |

### ⚠️ Elevated Energy (3 structures)
| PDB | Family | Atoms | Engine PE | Issue |
|-----|--------|-------|-----------|-------|
| 3SQQ | Marburg | 4,705 | 8.10e+10 | Multi-domain |
| 1HXY | HIV-1 | 9,444 | 3.24e+10 | Multi-chain envelope |
| 3HEC | Bacterial | 5,351 | 4.83e+11 | Enzyme flexibility |

### ❌ High Energy (7 structures)
| PDB | Family | Atoms | Engine PE | Issue |
|-----|--------|-------|-----------|-------|
| 4Z0J | Norovirus | 7,798 | 6.52e+12 | Capsid assembly |
| 2F4J | Human | 4,608 | 7.78e+14 | Kinase dynamics |
| 6M0J | SARS-CoV-2 | 12,510 | 6.84e+13 | RBD + ACE2 interface |
| 2VWD | Nipah | 12,926 | 2.86e+14 | Homo-tetramer |
| 4J1G | Influenza | 15,799 | 2.15e+14 | HA trimer |
| 4B7Q | Influenza | 23,312 | 1.25e+15 | NA tetramer |
| 5IRE | Zika | 26,297 | 1.70e+14 | Envelope hexamer |

---

## BY TIER

### Tier 1: Pandemic (7 structures)
| PDB | Family | Atoms | Status | Target Type |
|-----|--------|-------|--------|-------------|
| 6LU7 | SARS-CoV-2 | 4,730 | ✅ READY | Main protease (drug) |
| 1HXY | HIV-1 | 9,444 | ⚠️ ELEVATED | Envelope gp120 (vaccine) |
| 6M0J | SARS-CoV-2 | 12,510 | ❌ HIGH | Spike RBD (vaccine) |
| 2VWD | Nipah | 12,926 | ❌ HIGH | Attachment G (therapeutic) |
| 4J1G | Influenza | 15,799 | ❌ HIGH | Hemagglutinin (vaccine) |
| 4B7Q | Influenza | 23,312 | ❌ HIGH | Neuraminidase (drug) |
| 5IRE | Zika | 26,297 | ❌ HIGH | Envelope (vaccine) |

### Tier 2: Emerging (3 structures)
| PDB | Family | Atoms | Status | Target Type |
|-----|--------|-------|--------|-------------|
| 4QWO | Lassa | 4,148 | ✅ READY | Glycoprotein (vaccine) |
| 3SQQ | Marburg | 4,705 | ⚠️ ELEVATED | VP35 (immune evasion) |
| 4Z0J | Norovirus | 7,798 | ❌ HIGH | Capsid (vaccine) |

### Tier 3: Druggable/Benchmark (4 structures)
| PDB | Family | Atoms | Status | Target Type |
|-----|--------|-------|--------|-------------|
| 1L2Y | Synthetic | 304 | ✅ READY | Trp-cage (benchmark) |
| 1AKE | Human | 6,682 | ✅ READY | Adenylate kinase (cryptic) |
| 3HEC | Bacterial | 5,351 | ⚠️ ELEVATED | Enzyme |
| 2F4J | Human | 4,608 | ❌ HIGH | Kinase |

---

## BY SIZE CLASS

### Tiny (<1,000 atoms) - 1 structure
| PDB | Family | Atoms | Status |
|-----|--------|-------|--------|
| 1L2Y | Synthetic | 304 | ✅ READY |

### Small (1,000-5,000 atoms) - 5 structures
| PDB | Family | Atoms | Status |
|-----|--------|-------|--------|
| 4QWO | Lassa | 4,148 | ✅ READY |
| 2F4J | Human | 4,608 | ❌ HIGH |
| 3SQQ | Marburg | 4,705 | ⚠️ ELEVATED |
| 6LU7 | SARS-CoV-2 | 4,730 | ✅ READY |
| 3HEC | Bacterial | 5,351 | ⚠️ ELEVATED |

### Medium (5,000-10,000 atoms) - 3 structures
| PDB | Family | Atoms | Status |
|-----|--------|-------|--------|
| 1AKE | Human | 6,682 | ✅ READY |
| 4Z0J | Norovirus | 7,798 | ❌ HIGH |
| 1HXY | HIV-1 | 9,444 | ⚠️ ELEVATED |

### Large (10,000-20,000 atoms) - 3 structures
| PDB | Family | Atoms | Status |
|-----|--------|-------|--------|
| 6M0J | SARS-CoV-2 | 12,510 | ❌ HIGH |
| 2VWD | Nipah | 12,926 | ❌ HIGH |
| 4J1G | Influenza | 15,799 | ❌ HIGH |

### Extra Large (>20,000 atoms) - 2 structures
| PDB | Family | Atoms | Status |
|-----|--------|-------|--------|
| 4B7Q | Influenza | 23,312 | ❌ HIGH |
| 5IRE | Zika | 26,297 | ❌ HIGH |

---

## BY VIRUS FAMILY

### Coronavirus (2 structures)
| PDB | Protein | Atoms | Status | Use |
|-----|---------|-------|--------|-----|
| 6LU7 | Main Protease | 4,730 | ✅ READY | Paxlovid target |
| 6M0J | Spike RBD | 12,510 | ❌ HIGH | Vaccine target |

### Orthomyxovirus/Influenza (2 structures)
| PDB | Protein | Atoms | Status | Use |
|-----|---------|-------|--------|-----|
| 4J1G | Hemagglutinin | 15,799 | ❌ HIGH | Vaccine target |
| 4B7Q | Neuraminidase | 23,312 | ❌ HIGH | Drug resistance |

### Filovirus (1 structure)
| PDB | Protein | Atoms | Status | Use |
|-----|---------|-------|--------|-----|
| 3SQQ | VP35 | 4,705 | ⚠️ ELEVATED | Immune evasion |

### Arenavirus (1 structure)
| PDB | Protein | Atoms | Status | Use |
|-----|---------|-------|--------|-----|
| 4QWO | Glycoprotein | 4,148 | ✅ READY | Vaccine target |

### Paramyxovirus (1 structure)
| PDB | Protein | Atoms | Status | Use |
|-----|---------|-------|--------|-----|
| 2VWD | Attachment G | 12,926 | ❌ HIGH | Therapeutic |

### Flavivirus (1 structure)
| PDB | Protein | Atoms | Status | Use |
|-----|---------|-------|--------|-----|
| 5IRE | Envelope | 26,297 | ❌ HIGH | Vaccine target |

### Retrovirus (1 structure)
| PDB | Protein | Atoms | Status | Use |
|-----|---------|-------|--------|-----|
| 1HXY | Envelope gp120 | 9,444 | ⚠️ ELEVATED | Vaccine target |

### Calicivirus (1 structure)
| PDB | Protein | Atoms | Status | Use |
|-----|---------|-------|--------|-----|
| 4Z0J | Capsid | 7,798 | ❌ HIGH | Vaccine target |

### Human/Non-viral (3 structures)
| PDB | Protein | Atoms | Status | Use |
|-----|---------|-------|--------|-----|
| 1L2Y | Trp-cage | 304 | ✅ READY | Benchmark |
| 1AKE | Adenylate kinase | 6,682 | ✅ READY | Cryptic sites |
| 2F4J | Kinase | 4,608 | ❌ HIGH | Drug target |
| 3HEC | Enzyme | 5,351 | ⚠️ ELEVATED | Drug target |

---

## BY DOMAIN USE

### Vaccine Development (6 structures)
| PDB | Family | Status | Notes |
|-----|--------|--------|-------|
| 4QWO | Lassa | ✅ READY | Emerging threat |
| 1HXY | HIV-1 | ⚠️ ELEVATED | Cryptic epitopes |
| 6M0J | SARS-CoV-2 | ❌ HIGH | RBD immunogen |
| 4J1G | Influenza | ❌ HIGH | HA immunogen |
| 5IRE | Zika | ❌ HIGH | Envelope immunogen |
| 4Z0J | Norovirus | ❌ HIGH | Capsid immunogen |

### Drug Discovery (5 structures)
| PDB | Family | Status | Target |
|-----|--------|--------|--------|
| 6LU7 | SARS-CoV-2 | ✅ READY | Protease inhibitor |
| 1AKE | Human | ✅ READY | Cryptic pocket |
| 3HEC | Bacterial | ⚠️ ELEVATED | Enzyme inhibitor |
| 2F4J | Human | ❌ HIGH | Kinase inhibitor |
| 4B7Q | Influenza | ❌ HIGH | NA inhibitor |

### Escape Mutation Mapping (3 structures)
| PDB | Family | Status | Surveillance Use |
|-----|--------|--------|------------------|
| 4B7Q | Influenza | ❌ HIGH | Oseltamivir resistance |
| 6M0J | SARS-CoV-2 | ❌ HIGH | Antibody escape |
| 2VWD | Nipah | ❌ HIGH | Therapeutic escape |

### Benchmark/Methods (2 structures)
| PDB | Family | Status | Use |
|-----|--------|--------|-----|
| 1L2Y | Synthetic | ✅ READY | Folding benchmark |
| 1AKE | Human | ✅ READY | Apo-holo benchmark |

---

## SUMMARY

### Overall Statistics
- Total structures: 14
- Production ready: 4 (29%)
- Elevated energy: 3 (21%)
- High energy: 7 (50%)

### Topology Preparation: 100% SUCCESS
All 14 structures now have correct OpenMM minimization energies (negative values) after glycan preprocessing.

### Engine Performance by Size
| Size Class | Total | Ready | Success Rate |
|------------|-------|-------|--------------|
| Tiny (<1K) | 1 | 1 | 100% |
| Small (1-5K) | 5 | 2 | 40% |
| Medium (5-10K) | 3 | 1 | 33% |
| Large (10-20K) | 3 | 0 | 0% |
| XL (>20K) | 2 | 0 | 0% |

### Ready for Production
1. **1L2Y** - Trp-cage miniprotein (folding benchmark)
2. **4QWO** - Lassa virus glycoprotein (emerging threat vaccine)
3. **6LU7** - SARS-CoV-2 main protease (Paxlovid drug target)
4. **1AKE** - Adenylate kinase (cryptic site benchmark)
