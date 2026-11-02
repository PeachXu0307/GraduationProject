import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'/'graph'))
from check_graph import audit

def test_graph_mapping_and_line_physics():
    r=audit(ROOT/'data/graphs/ieee13',ROOT/'data/raw/ieee13')
    assert r['graphs']==900
    assert r['buses']==16 and r['branches']==17 and r['directed_edges']==34
    assert r['phase_nodes']==41
    assert r['max_line_current_relative_error']<1e-5
