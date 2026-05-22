"""Standard PRISM-DSTW boundary exceptions."""

from __future__ import annotations


class FatalBoundaryError(Exception):
    """Raised when air-gap, schema, or physical validation fails."""

