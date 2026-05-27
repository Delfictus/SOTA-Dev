"""xTB command adapter contract.

The hardened release tests verify the interface without requiring the xTB
binary. Production calls use :meth:`run_xtb` and fail closed if the executable
is unavailable or output parsing fails.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class XTBRewardAdapter:
    executable: str = "xtb"
    gfn: int = 2
    electronic_temperature: int = 300

    def run_xtb(self, coord_xyz: Path) -> str:
        result = subprocess.run(
            [
                self.executable,
                coord_xyz.as_posix(),
                "--gfn",
                str(self.gfn),
                "--etemp",
                str(self.electronic_temperature),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "xTB command failed")
        return result.stdout

    def compute_homo_lumo_gap(self, coord_xyz: Path) -> float:
        output = self.run_xtb(coord_xyz)
        return self.parse_homo_lumo_gap(output)

    def compute_electron_affinity(self, coord_xyz: Path) -> float:
        output = self.run_xtb(coord_xyz)
        return self.parse_electron_affinity(output)

    @staticmethod
    def parse_homo_lumo_gap(output: str) -> float:
        for pattern in (r"HOMO-LUMO GAP\s+([-+0-9.eE]+)", r"HL-Gap\s+([-+0-9.eE]+)"):
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return float(match.group(1))
        raise ValueError("could not parse HOMO-LUMO gap from xTB output")

    @staticmethod
    def parse_electron_affinity(output: str) -> float:
        match = re.search(r"electron affinity\s+([-+0-9.eE]+)", output, re.IGNORECASE)
        if match:
            return float(match.group(1))
        raise ValueError("could not parse electron affinity from xTB output")
