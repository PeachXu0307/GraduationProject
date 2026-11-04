import json, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'/'physics'))
from partial_ac import partial_ac_residual


def test_partial_residual_has_gradient_and_explicit_scope():
    top=json.loads((ROOT/'data/graphs/ieee13/topology.json').read_text(encoding='utf-8'))
    sample=json.loads((ROOT/'data/raw/ieee13/samples.jsonl').read_text(encoding='utf-8').splitlines()[0])
    g=torch.load(ROOT/'data/graphs/ieee13/graphs.pt',weights_only=False)[0]
    pred=(g.y.clone().detach()).requires_grad_(True)
    bases=torch.tensor([b['v_base_ln_kv']*1000 for b in sample['buses'][0:0]],dtype=torch.float32)
    bybus={}
    for b in sample['buses']:
        bybus.setdefault(b['bus'],b['v_base_ln_kv']*1000)
    bases=torch.tensor([bybus[x] for x in top['bus_names']],dtype=torch.float32)
    pred_pu=pred / bases[:, None]
    out=partial_ac_residual(pred_pu,sample,top,bases)
    assert out['residual_a'].shape==(41,)
    assert out['covered_mask'].shape==(41,)
    assert 'transformer' in out['omitted_devices'] and 'vsource' in out['omitted_devices']
    assert torch.isfinite(out['residual_a']).all()
    out['residual_rms_a'].backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_solved_voltage_has_small_residual_on_covered_nodes():
    top=json.loads((ROOT/'data/graphs/ieee13/topology.json').read_text(encoding='utf-8'))
    rows=(ROOT/'data/raw/ieee13/samples.jsonl').read_text(encoding='utf-8').splitlines()
    for row in (rows[0], rows[-1]):
        sample=json.loads(row)
        bybus={b['bus']:b['v_base_ln_kv']*1000 for b in sample['buses']}
        volts={(b['bus'],b['node']):complex(*b['v_re_im_v']) for b in sample['buses']}
        bases=torch.tensor([bybus[name] for name in top['bus_names']],dtype=torch.float64)
        pred=torch.tensor([[[volts.get((name,p),0j).real/bybus[name],
                             volts.get((name,p),0j).imag/bybus[name]]
                            for p in (1,2,3)] for name in top['bus_names']],dtype=torch.float64).reshape(-1,6)
        out=partial_ac_residual(pred,sample,top,bases)
        covered=out['covered_mask']
        assert covered.sum().item()>0
        assert out['residual_a'][covered].abs().max().item()<0.5
        for name,mask in zip(top['phase_node_order'],covered.tolist()):
            if name.startswith(('sourcebus.','650.','rg60.','633.','634.','611.','675.')):
                assert not mask
