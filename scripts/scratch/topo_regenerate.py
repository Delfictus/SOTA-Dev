#!/usr/bin/env python3
"""Topological order for PRISM-DSTW n80 Phase 1-4 extraction."""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class DagNode:
    name: str
    script: str
    output: str


NODES = (
    DagNode("noise_floor_extractor", "scripts/prism_noise_floor_extractor.py", "stream_snr_masks.parquet"),
    DagNode("bocpd_survival_extractor", "scripts/prism_bocpd_extractor.py", "bocpd_survival_regimes.parquet"),
    DagNode("protocol_state_extractor", "scripts/prism_protocol_state_extractor.py", "autonomous_steering_tensor.parquet"),
    DagNode("kcc_decoder", "scripts/prism_kcc_decoder.py", "kcc_residue_fields.parquet"),
    DagNode("spike_event_integrator", "scripts/prism_spike_event_integrator.py", "spike_events_snr_masked.parquet"),
    DagNode("adaptive_dt_extractor", "scripts/prism_adaptive_dt_extractor.py", "kinetic_strain_events.parquet"),
    DagNode("aromatic_kinematics", "crates/prism-nhs/src/bin/aromatic_kinematics.rs", "aromatic_reorganization_tensor.parquet"),
    DagNode("mechanical_load", "crates/prism-nhs/src/bin/mechanical_load.rs", "mechanical_load_network.parquet"),
    DagNode("signal_grid_differential", "crates/prism-nhs/src/bin/signal_grid_differential.rs", "signal_grid_variance_channel.parquet"),
)

EDGES = (
    ("noise_floor_extractor", "spike_event_integrator"),
    ("bocpd_survival_extractor", "adaptive_dt_extractor"),
    ("protocol_state_extractor", "aromatic_kinematics"),
    ("protocol_state_extractor", "signal_grid_differential"),
)


def topological_order() -> list[str]:
    node_names = {node.name for node in NODES}
    incoming: dict[str, int] = {name: 0 for name in node_names}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target in EDGES:
        if source not in node_names or target not in node_names:
            raise ValueError(f"edge references unknown node: {source}->{target}")
        outgoing[source].append(target)
        incoming[target] += 1
    queue: deque[str] = deque(node.name for node in NODES if incoming[node.name] == 0)
    order: list[str] = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for target in outgoing[current]:
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    if len(order) != len(NODES):
        raise ValueError("cycle detected in Phase 1-4 extraction DAG")
    return order


def main() -> int:
    order = topological_order()
    sys.stdout.write("TOPOLOGICAL_ORDER\n")
    for index, name in enumerate(order, start=1):
        node = next(item for item in NODES if item.name == name)
        sys.stdout.write(f"{index}. {node.name} -> {node.script} -> {node.output}\n")
    sys.stdout.write("DEPENDENCY_EDGES\n")
    for source, target in EDGES:
        sys.stdout.write(f"{source} -> {target}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
