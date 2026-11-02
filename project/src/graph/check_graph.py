"""Audit phase graph tensors and line-only Ybus against OpenDSS records."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]

def audit(graph_dir: Path, raw_dir: Path) -> dict:
    top=json.loads((graph_dir/'topology.json').read_text(encoding='utf-8'))
    graphs=torch.load(graph_dir/'graphs.pt',weights_only=False)
    rows=[json.loads(s) for s in (raw_dir/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(graphs)==len(rows)
    names=top['bus_names']; n=len(names); branches=top['branches']; b=len(branches)
    incidence=np.asarray(top['incidence'],dtype=np.float64)
    adjacency=np.asarray(top['adjacency'],dtype=np.int8)
    yl=np.asarray(top['line_ybus_re_s'])+1j*np.asarray(top['line_ybus_im_s'])
    assert incidence.shape==(b,n)
    assert adjacency.shape==(n,n) and np.array_equal(adjacency,adjacency.T)
    assert yl.shape==(41,41) and len(top['phase_node_order'])==41
    assert np.allclose(yl,yl.T,atol=1e-9)
    assert np.all((incidence==-1).sum(1)==1) and np.all((incidence==1).sum(1)==1)
    max_line_irel=0.0
    for graph,row in zip(graphs,rows):
        assert graph.sample_id==row['sample_id']
        assert graph.x.shape==(n,13) and graph.y.shape==(n,6)
        assert graph.edge_index.shape==(2,2*b) and graph.edge_attr.shape==(2*b,28)
        assert graph.branch_id.shape==(2*b,)
        assert torch.isfinite(graph.x).all() and torch.isfinite(graph.y).all()
        assert torch.equal(graph.x[:,:3],graph.phase_mask)
        actual={(x['bus'],x['node']) for x in row['buses']}
        expected={(names[i],p+1) for i in range(n) for p in range(3) if graph.phase_mask[i,p]}
        assert actual==expected
        byname={e['name']:e for e in row['elements']}
        lines={e['name']:e for e in row['lines']}
        voltage={(x['bus'],x['node']):complex(*x['v_re_im_v']) for x in row['buses']}
        # Topology and equipment type must be identical across all samples.
        current_branches=[e for e in row['elements'] if e['kind'] in {'line','transformer'}]
        assert [e['name'] for e in current_branches]==[e['name'] for e in branches]
        for j,e in enumerate(branches):
            raw=byname[e['name']]
            assert [t['bus'] for t in raw['terminals']]==[e['u'],e['v']]
            assert raw['terminals'][0]['nodes']==e['u_nodes']
            assert raw['terminals'][1]['nodes']==e['v_nodes']
            assert graph.branch_id[2*j:2*j+2].tolist()==[j,j]
            u,v=names.index(e['u']),names.index(e['v'])
            assert graph.edge_index[:,2*j:2*j+2].T.tolist()==[[u,v],[v,u]]
            if e['kind']!='line': continue
            meta=lines[e['name']]
            order=[p for p in e['u_nodes'] if p in {1,2,3}]
            assert len(order)==len(e['v_nodes'])==int(raw['num_phases'])
            z=(np.asarray(meta['r_matrix_per_length_ohm'])+1j*np.asarray(meta['x_matrix_per_length_ohm']))*meta['length']
            c=np.asarray(meta['c_matrix_per_length_nf'])*meta['length']*1e-9
            y=np.linalg.inv(z)
            vs=np.asarray([voltage[(e['u'],p)] for p in order])
            vr=np.asarray([voltage[(e['v'],p)] for p in e['v_nodes']])
            # Pi-line end current, using 60 Hz and the OpenDSS per-length matrix.
            ish=0.5j*2*np.pi*60*c@vs
            estimate=y@(vs-vr)+ish
            observed=np.asarray([complex(*q) for q in raw['terminals'][0]['current_re_im_a']])
            rel=np.linalg.norm(estimate-observed)/max(np.linalg.norm(observed),1.0)
            max_line_irel=max(max_line_irel,float(rel))
    return {'graphs':len(graphs),'buses':n,'branches':b,'directed_edges':2*b,
            'phase_nodes':int(graphs[0].phase_mask.sum()),'max_line_current_relative_error':max_line_irel}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--graphs',type=Path,default=ROOT/'data/graphs/ieee13')
    p.add_argument('--raw',type=Path,default=ROOT/'data/raw/ieee13')
    a=p.parse_args(); print(json.dumps(audit(a.graphs,a.raw),indent=2))
if __name__=='__main__': main()
