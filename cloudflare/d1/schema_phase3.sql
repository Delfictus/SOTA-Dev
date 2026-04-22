-- Phase 3 schema extension: JSON blob columns for high-dimensional feature vectors
ALTER TABLE residue_features ADD COLUMN physics_features TEXT;
ALTER TABLE residue_features ADD COLUMN nma_features TEXT;
ALTER TABLE residue_features ADD COLUMN perturbed_nma_features TEXT;
