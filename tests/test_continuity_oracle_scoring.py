from pathlib import Path
import subprocess


def test_oracle_help_exposes_continuity_flags() -> None:
    oracle_binary = Path("target/release/oracle_scorer")
    if not oracle_binary.exists():
        oracle_binary = Path("campaigns/glp1r_aleniglipron/track_b_chronological/runtime/bin/oracle_scorer")
    result = subprocess.run(
        [str(oracle_binary), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "--continuity-admissibility" in result.stdout
    assert "--nma-continuity-map" in result.stdout
    assert "--hydration-continuity-map" in result.stdout
    assert "--thermodynamic-continuity-map" in result.stdout
