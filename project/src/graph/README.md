# Stage 2: three-phase graph representation

## Mathematical choice

The recommended representation is a **bus graph with phase slots**. Each
physical bus is one graph node. Its three phase values occupy fixed A/B/C
slots and `phase_mask[i,p]` is one only when that phase exists. This keeps the
graph topology aligned with OpenDSS and avoids treating absent phases as zero
voltage. A physical line, transformer, or regulator is one physical branch
with a pair of directed message edges. `branch_id` maps the two directed
edges back to the same device.

For a line with conductor order \(\phi_1,\ldots,\phi_m\), the graph keeps
the full self and mutual matrices:

\[
Z_{ij}=R_{ij}+jX_{ij}, \qquad
I_{ij}=Z_{ij}^{-1}(V_i-V_j)+\frac{j\omega C_{ij}}{2}V_i.
\]

The implementation stores the R and X matrices in A/B/C slots, the line
length, OpenDSS units code, and diagonal C entries. The matrices in the raw
JSON are **per-length values**; a later physics layer must multiply by the
line length. A/B/C slot placement is based on the actual terminal conductor
numbers, not on the number of conductors alone.

Transformers and regulators are kept as type `transformer` branches. Their
graph edge has the device type and phase mask, but zero line-matrix slots.
This is deliberate: a transformer connection and tap model cannot be
replaced by a line impedance matrix. Stage 3 must use their exact port model
or OpenDSS-derived `Yprim` when computing a complete physical residual.

An alternative is phase-node expansion, where each bus-phase is a graph node
and mutual coupling becomes cross-phase edges. That representation is useful
for explicit phase-level message passing and missing-phase nodes, but it
requires more careful treatment of delta loads, transformers, and aggregation
back to physical buses. It is retained as a possible ablation, not the main
implementation.

## Tensor dimensions

For the default IEEE 13 data:

| Tensor | Shape | Meaning |
|---|---:|---|
| `x` | `(16, 13)` | phase mask; approximate P/Q input features; source, load, DG, capacitor flags |
| `phase_mask` | `(16, 3)` | valid A/B/C phase slots |
| `y` | `(16, 6)` | Re/Im voltage target for A/B/C; invalid slots are zero and masked |
| `edge_index` | `(2, 34)` | 17 physical branches represented in both directions |
| `edge_attr` | `(34, 28)` | type, phase mask, length, R/X/C slots, unit code |
| `branch_id` | `(34,)` | maps directed edges to physical branches |
| incidence | `(17, 16)` | -1 at the first terminal bus, +1 at the second |
| adjacency | `(16, 16)` | symmetric physical-bus connectivity |
| line Ybus | `(41, 41)` complex | line-only three-phase nodal admittance in siemens |

The input P/Q features use equal allocation for a multi-phase load as a
convenient graph input proxy. This is **not** the exact delta-load injection;
the original terminal records remain in Stage 1 JSONL and must be used for
the AC physics layer. The first GNN target is node complex voltage. Branch
current and P/Q will be calculated from the predicted voltages and exact
device models rather than independently asserted by the graph.

## Generate and verify

From `project/`:

```powershell
.venv\Scripts\python.exe src\graph\build_graph.py
.venv\Scripts\python.exe src\graph\check_graph.py
.venv\Scripts\python.exe -m pytest tests\test_stage2_graph.py -q -p no:faulthandler -p no:cacheprovider
```

The graph files are written under `data/graphs/ieee13/`: `graphs.pt` contains
900 PyG `Data` objects and `topology.json` contains bus names, branch metadata
and physical incidence, adjacency and line-only Ybus matrices. `phase_node_order`
defines the Ybus row/column order. The line Ybus includes full R/X mutual
coupling and shunt C matrix, while the GNN edge feature stores C diagonal
entries only. The check verifies every sample's phase
mask, branch endpoint and conductor ordering. It also reconstructs each
line's pi-model sending-end current from the recorded voltages and compares it
with OpenDSS. The current default result is 900 graphs, 41 valid phase nodes,
and maximum relative line-current error about `2.71e-8`.

This is a topology and line-device audit; it is not yet a complete AC Ybus
audit because transformer, regulator, source, load, capacitor, and generator
port models are not all represented by line matrices. That complete physical
residual belongs to Stage 3.
