//! Pocket druggability helpers (placeholder; scoring lives in scoring module)

use crate::pocket::properties::Pocket;

pub fn update_basic_pocket_stats(pocket: &mut Pocket) {
    if pocket.atom_indices.is_empty() {
        return;
    }
    // Placeholder: volume/enclosure to be computed by geometry module later
    pocket.volume = pocket.volume.max(0.0);
}
