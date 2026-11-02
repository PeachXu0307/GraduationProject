"""Independently check branch Yprim and load equations on Stage 1 scenarios.

This is a read-only audit. It recompiles the source case per sampled scenario,
reapplies its *input* settings, solves, and compares calculated port currents
with OpenDSS. No solved current is used to calculate the proposed current.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import dss

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'/'simulation'))
from generate_ieee13 import CIRCUIT, apply_scenario

def complex_pairs(x):
    a=np.asarray(x,dtype=float).reshape(-1,2)
    return a[:,0]+1j*a[:,1]

def rel_error(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1.0))

def bus_voltage(circuit,bus):
    circuit.SetActiveBus(bus)
    b=circuit.ActiveBus
    return {int(n):v for n,v in zip(b.Nodes,complex_pairs(b.Voltages))}

def element_voltages(circuit,element):
    nc=int(element.NumConductors)
    values=[]
    for t,bus_spec in enumerate(element.BusNames):
        vm=bus_voltage(circuit,bus_spec.split('.')[0])
        values.extend(vm.get(int(n),0j) for n in element.NodeOrder[t*nc:(t+1)*nc])
    return np.asarray(values,dtype=complex)

def yprim_current(circuit,name):
    circuit.SetActiveElement(name)
    e=circuit.ActiveCktElement
    n=int(e.NumConductors*e.NumTerminals)
    y=complex_pairs(e.Yprim).reshape(n,n)
    v=element_voltages(circuit,e)
    return y@v,complex_pairs(e.Currents)

def load_current(circuit,name):
    loads=circuit.Loads; loads.Name=name
    circuit.SetActiveElement('Load.'+name);e=circuit.ActiveCktElement
    nodes=[int(x) for x in e.NodeOrder]
    v=element_voltages(circuit,e)
    delta=bool(loads.IsDelta); phases=int(loads.Phases); model=int(loads.Model)
    s=complex(loads.kW,loads.kvar)*1000
    if delta:
        if phases==1:
            branch_v=np.asarray([v[0]-v[1]])
            mapping=np.asarray([[1],[-1]])
        elif phases==3:
            branch_v=np.asarray([v[0]-v[1],v[1]-v[2],v[2]-v[0]])
            mapping=np.asarray([[1,0,-1],[-1,1,0],[0,-1,1]])
        else: raise ValueError(f'Unsupported delta phase count: {name}')
    else:
        branch_v=v[:phases]
        mapping=np.eye(phases)
    nominal=loads.kV*1000/(np.sqrt(3) if not delta and phases==3 else 1)
    power_per_branch=s/len(branch_v)
    if model==1:
        ib=np.conj(power_per_branch/branch_v)
    elif model==2:
        ib=np.conj(power_per_branch)/(nominal**2)*branch_v
    elif model==5:
        ib=np.conj(power_per_branch)/nominal*branch_v/np.abs(branch_v)
    else: raise ValueError(f'Unsupported load model {model}: {name}')
    # OpenDSS switches Model 1 and 5 to constant impedance outside their
    # Vminpu/Vmaxpu band, matching the current continuously at the boundary.
    if model in {1,5}:
        magnitude=np.abs(branch_v)
        lower=loads.Vminpu*nominal
        upper=loads.Vmaxpu*nominal
        boundary=np.where(magnitude<lower,lower,np.where(magnitude>upper,upper,magnitude))
        below_or_above=(magnitude<lower)|(magnitude>upper)
        if model==1:
            fallback=np.conj(power_per_branch)/(boundary**2)*branch_v
        else:
            fallback=np.conj(power_per_branch)/(nominal*boundary)*branch_v
        ib=np.where(below_or_above,fallback,ib)
    calc=mapping@ib
    obs=complex_pairs(e.Currents)[:len(calc)]
    return calc,obs,model,delta

def base_loads(circuit):
    out={}
    for name in circuit.Loads.AllNames:
        circuit.Loads.Name=name
        circuit.SetActiveElement('Load.'+name)
        out[name]={'kw':float(circuit.Loads.kW),'kvar':float(circuit.Loads.kvar),
                   'bus':circuit.ActiveCktElement.BusNames[0],
                   'model':int(circuit.Loads.Model),'phases':int(circuit.Loads.Phases),
                   'is_delta':bool(circuit.Loads.IsDelta)}
    return out

def audit_one(params):
    dss.DSS.Text.Command=f'Compile "{CIRCUIT}"'
    circuit=dss.DSS.ActiveCircuit
    apply_scenario(circuit,params,base_loads(circuit))
    circuit.Solution.Solve()
    if not circuit.Solution.Converged: raise RuntimeError('Audit scenario did not converge')
    rows=[]
    for name in circuit.AllElementNames:
        circuit.SetActiveElement(name);e=circuit.ActiveCktElement
        if not e.Enabled: continue
        kind=name.split('.')[0].lower()
        if kind in {'line','transformer','capacitor'}:
            calc,obs=yprim_current(circuit,name)
            rows.append({'element':name.lower(),'kind':kind,'method':'Yprim',
                         'relative_error':rel_error(calc,obs)})
        elif kind=='load':
            calc,obs,model,delta=load_current(circuit,name.split('.')[1])
            rows.append({'element':name.lower(),'kind':kind,'method':f'model{model}_{"delta" if delta else "wye"}',
                         'relative_error':rel_error(calc,obs)})
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,default=ROOT/'data/raw/ieee13')
    p.add_argument('--sample-count',type=int,default=12)
    p.add_argument('--out',type=Path,default=ROOT/'results/stage3_edge/port_audit.json')
    a=p.parse_args()
    records=[json.loads(x) for x in (a.raw/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    # Spread audits across ordinary and stress operating points.
    indices=np.linspace(0,len(records)-1,a.sample_count,dtype=int)
    rows=[]
    for i in indices:
        for r in audit_one(records[i]['scenario']): rows.append({'sample_id':records[i]['sample_id'],**r})
    summary={}
    for key in sorted(set(x['method'] for x in rows)):
        errors=[x['relative_error'] for x in rows if x['method']==key]
        summary[key]={'count':len(errors),'max_relative_error':max(errors),
                      'mean_relative_error':float(np.mean(errors))}
    report={'samples_audited':len(indices),'summary':summary,'records':rows}
    a.out=a.out.resolve();a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'samples_audited':len(indices),'summary':summary},indent=2))

if __name__=='__main__': main()
