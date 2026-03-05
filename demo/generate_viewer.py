#!/usr/bin/env python3
"""
PRISM-4D Interactive Binding Site Viewer Generator
Parses .pml + .binding_sites.pdb + source PDB → 3Dmol.js HTML
"""

import json
import re
import sys
import os
from pathlib import Path

def parse_pml(pml_path):
    pockets = {}
    with open(pml_path) as f:
        content = f.read()
    pocket_pattern = re.compile(r'# =+ Site (\d+) \((\w+)\) \[(\w+)\] =+')
    for match in pocket_pattern.finditer(content):
        site_num = int(match.group(1))
        pockets[site_num] = {
            'type': match.group(2),
            'druggable': match.group(3) == 'DRUGGABLE',
            'lining_residues': [], 'catalytic_residues': [],
            'aromatic_residues': [], 'hydrophobic_residues': [],
            'centroid': None,
        }
    select_pattern = re.compile(r'select pocket_(\d+)_(lining|catalytic|aromatic|hydrophobic),\s*(.+)')
    for match in select_pattern.finditer(content):
        pocket_num = int(match.group(1))
        res_type = match.group(2)
        residues = [int(r) for r in re.findall(r'resi (\d+)', match.group(3))]
        chain_match = re.search(r'chain (\w)', match.group(3))
        chain = chain_match.group(1) if chain_match else 'A'
        if pocket_num in pockets:
            pockets[pocket_num][f'{res_type}_residues'] = residues
            pockets[pocket_num]['chain'] = chain
    centroid_pattern = re.compile(r'pseudoatom pocket_(\d+)_center, pos=\[([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\]')
    for match in centroid_pattern.finditer(content):
        pocket_num = int(match.group(1))
        if pocket_num in pockets:
            pockets[pocket_num]['centroid'] = [float(match.group(2)), float(match.group(3)), float(match.group(4))]
    return pockets

def parse_binding_sites_pdb(pdb_path):
    scores = {}
    type_map = {'ACT': 'ActiveSite', 'CRY': 'Cryptic', 'ALO': 'Allosteric', 'UNK': 'Unknown'}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('HETATM'):
                site_idx = int(line.split()[1])
                resname = line.split()[2]
                scores[site_idx] = {
                    'type': type_map.get(resname, 'Unknown'),
                    'centroid': [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    'druggability': float(line[54:60]),
                    'quality': float(line[60:66]),
                }
    return scores

def generate_html(pockets, scores, pdb_content, pdb_id):
    pocket_data = []
    for site_num in sorted(pockets.keys()):
        p = pockets[site_num]
        s = scores.get(site_num, {})
        pocket_data.append({
            'id': site_num, 'type': p.get('type','Unknown'),
            'druggable': p.get('druggable',False), 'chain': p.get('chain','A'),
            'lining': p.get('lining_residues',[]), 'catalytic': p.get('catalytic_residues',[]),
            'aromatic': p.get('aromatic_residues',[]), 'hydrophobic': p.get('hydrophobic_residues',[]),
            'centroid': p.get('centroid',[0,0,0]),
            'druggability_score': s.get('druggability',0), 'quality_score': s.get('quality',0),
        })
    pdb_escaped = pdb_content.replace('\\','\\\\').replace('`','\\`').replace('${','\\${')
    pdata_json = json.dumps(pocket_data, indent=2)

    html = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PRISM-4D Viewer — ''' + pdb_id.upper() + '''</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
:root{--bg:#06060a;--bg2:#0c0c14;--card:#0f0f1a;--cyan:#00e5ff;--cyan-dim:#00a5b8;--green:#00ff88;--amber:#ffaa00;--red:#ff3366;--purple:#aa44ff;--text:#e8ecf0;--text2:#8892a0;--dim:#556070;--border:rgba(0,229,255,0.12);--border-active:rgba(0,229,255,0.4)}
*{box-sizing:border-box;margin:0;padding:0}
html{scrollbar-width:thin;scrollbar-color:var(--cyan-dim) var(--bg)}
body{font-family:'Space Grotesk',sans-serif;background:var(--bg);color:var(--text);overflow-x:hidden}
.top-bar{background:var(--cyan);color:var(--bg);text-align:center;padding:3px 0;font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;letter-spacing:4px;text-transform:uppercase}
header{padding:20px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;background:rgba(6,6,10,0.95)}
header h1{font-size:20px;font-weight:700;letter-spacing:1px}
header h1 span{color:var(--cyan)}
.header-meta{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--dim);letter-spacing:1px}
.layout{display:grid;grid-template-columns:320px 1fr;height:calc(100vh - 75px)}
.sidebar{background:var(--bg2);border-right:1px solid var(--border);overflow-y:auto;padding:16px}
.sidebar h2{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--cyan);margin-bottom:12px}
.pocket-card{background:var(--card);border:1px solid var(--border);border-radius:6px;margin-bottom:8px;cursor:pointer;transition:all 0.2s ease;overflow:hidden}
.pocket-card:hover{border-color:var(--border-active);box-shadow:0 0 20px rgba(0,229,255,0.06)}
.pocket-card.active{border-color:var(--cyan);box-shadow:0 0 24px rgba(0,229,255,0.12)}
.pocket-header{display:flex;align-items:center;justify-content:space-between;padding:10px 12px}
.pocket-header .name{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600}
.pocket-header .badge{font-family:'JetBrains Mono',monospace;font-size:9px;padding:2px 8px;border-radius:20px;letter-spacing:1px;text-transform:uppercase}
.badge-active{background:rgba(255,51,102,0.15);color:var(--red);border:1px solid rgba(255,51,102,0.3)}
.badge-cryptic{background:rgba(0,229,255,0.12);color:var(--cyan);border:1px solid rgba(0,229,255,0.3)}
.badge-allosteric{background:rgba(170,68,255,0.12);color:var(--purple);border:1px solid rgba(170,68,255,0.3)}
.badge-unknown{background:rgba(255,170,0,0.12);color:var(--amber);border:1px solid rgba(255,170,0,0.3)}
.pocket-scores{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:0 12px 10px}
.score{text-align:center;padding:6px;background:rgba(0,229,255,0.03);border:1px solid var(--border);border-radius:4px}
.score-label{font-family:'JetBrains Mono',monospace;font-size:8px;color:var(--dim);letter-spacing:1px;text-transform:uppercase}
.score-val{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;color:var(--cyan)}
.pocket-residues{padding:0 12px 10px;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--dim);line-height:1.6;display:none}
.pocket-card.active .pocket-residues{display:block}
.res-cat{color:#ff3366}.res-aro{color:#00ff88}.res-hyd{color:#ffaa00}
.controls{padding:12px;border-top:1px solid var(--border);display:flex;flex-wrap:wrap;gap:6px}
.ctrl-btn{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1px;padding:6px 12px;background:var(--card);border:1px solid var(--border);border-radius:4px;color:var(--text2);cursor:pointer;transition:all 0.2s ease;text-transform:uppercase}
.ctrl-btn:hover{border-color:var(--cyan);color:var(--cyan)}
.ctrl-btn.active{border-color:var(--cyan);color:var(--cyan);background:rgba(0,229,255,0.08)}
.viewer-wrap{position:relative;background:#000}
#viewer{width:100%;height:100%}
.viewer-legend{position:absolute;bottom:16px;left:16px;background:rgba(6,6,10,0.85);backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:6px;padding:12px 16px;font-family:'JetBrains Mono',monospace;font-size:10px;z-index:10}
.legend-item{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.legend-dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.viewer-info{position:absolute;top:16px;right:16px;background:rgba(6,6,10,0.85);backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:6px;padding:10px 14px;font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text2);z-index:10;max-width:280px}
.viewer-info .pocket-name{color:var(--cyan);font-weight:600;font-size:13px;margin-bottom:4px}
@media(max-width:900px){.layout{grid-template-columns:1fr;grid-template-rows:auto 50vh}.sidebar{max-height:40vh}}
</style>
</head>
<body>
<div class="top-bar">PRISM-4D · Neuromorphic Binding Site Detection</div>
<header>
<h1>PRISM<span>-4D</span> Viewer</h1>
<div class="header-meta">''' + pdb_id.upper() + ''' · ''' + str(len(pocket_data)) + ''' BINDING SITES DETECTED</div>
</header>
<div class="layout">
<div class="sidebar">
<h2>Detected Pockets</h2>
<div id="pocket-list"></div>
<div class="controls">
<button class="ctrl-btn active" onclick="toggleProtein(this)">Protein</button>
<button class="ctrl-btn active" onclick="toggleSurfaces(this)">Surfaces</button>
<button class="ctrl-btn" onclick="toggleCentroids(this)">Centroids</button>
<button class="ctrl-btn" onclick="toggleLabels(this)">Labels</button>
<button class="ctrl-btn" onclick="showAll()">Show All</button>
<button class="ctrl-btn" onclick="hideAll()">Hide All</button>
<button class="ctrl-btn" onclick="resetView()">Reset</button>
</div>
</div>
<div class="viewer-wrap">
<div id="viewer"></div>
<div class="viewer-legend">
<div class="legend-item"><span class="legend-dot" style="background:#ff3366"></span> Catalytic</div>
<div class="legend-item"><span class="legend-dot" style="background:#00ff88"></span> Aromatic</div>
<div class="legend-item"><span class="legend-dot" style="background:#ffaa00"></span> Hydrophobic</div>
<div class="legend-item"><span class="legend-dot" style="background:rgba(100,140,200,0.5)"></span> Pocket Surface</div>
</div>
<div class="viewer-info" id="pocket-info" style="display:none">
<div class="pocket-name" id="info-name"></div>
<div id="info-detail"></div>
</div>
</div>
</div>
<script>
const POCKETS = ''' + pdata_json + ''';
const TYPE_COLORS = {ActiveSite:0xff3366, Cryptic:0x00e5ff, Allosteric:0xaa44ff, Unknown:0xffaa00};
const PDB_DATA = `''' + pdb_escaped + '''`;
let viewer, showProtein=true, showSurfaces=true, showCentroids=false, showLabels=false;
let activePocket=null, pocketSurfaces={}, centroidSpheres=[], labelSprites=[];

function initViewer(){
    const el=document.getElementById('viewer');
    viewer=$3Dmol.createViewer(el,{backgroundColor:0x06060a,antialias:true});
    viewer.addModel(PDB_DATA,'pdb');
    viewer.setStyle({},{cartoon:{color:'gray',opacity:0.7}});
    POCKETS.forEach(p=>{
        const c=p.chain||'A';
        if(p.lining.length) viewer.setStyle({chain:c,resi:p.lining},{stick:{radius:0.12,color:'gray'},cartoon:{color:'gray',opacity:0.7}});
        if(p.catalytic.length) viewer.setStyle({chain:c,resi:p.catalytic},{stick:{radius:0.15,color:0xff3366},cartoon:{color:'gray',opacity:0.7}});
        if(p.aromatic.length) viewer.setStyle({chain:c,resi:p.aromatic},{stick:{radius:0.15,color:0x00ff88},cartoon:{color:'gray',opacity:0.7}});
        if(p.hydrophobic.length) viewer.setStyle({chain:c,resi:p.hydrophobic},{stick:{radius:0.15,color:0xffaa00},cartoon:{color:'gray',opacity:0.7}});
        if(p.lining.length){
            const surf=viewer.addSurface($3Dmol.SurfaceType.VDW,{opacity:0.25,color:TYPE_COLORS[p.type]||0xffaa00},{chain:c,resi:p.lining},{chain:c,resi:p.lining});
            pocketSurfaces[p.id]=surf;
        }
    });
    viewer.zoomTo(); viewer.render(); buildPocketList();
}

function buildPocketList(){
    const list=document.getElementById('pocket-list');
    list.innerHTML='';
    POCKETS.forEach(p=>{
        const bc=p.type==='ActiveSite'?'badge-active':p.type==='Cryptic'?'badge-cryptic':p.type==='Allosteric'?'badge-allosteric':'badge-unknown';
        const card=document.createElement('div');
        card.className='pocket-card'; card.dataset.pocketId=p.id;
        card.innerHTML=`<div class="pocket-header"><span class="name">Site ${p.id}</span><span class="badge ${bc}">${p.type}</span></div><div class="pocket-scores"><div class="score"><div class="score-label">Quality</div><div class="score-val">${p.quality_score.toFixed(1)}</div></div><div class="score"><div class="score-label">Drug.</div><div class="score-val">${p.druggability_score.toFixed(2)}</div></div></div><div class="pocket-residues">${p.catalytic.length?'<span class="res-cat">CAT:</span> '+p.catalytic.join(', ')+'<br>':''}${p.aromatic.length?'<span class="res-aro">ARO:</span> '+p.aromatic.join(', ')+'<br>':''}${p.hydrophobic.length?'<span class="res-hyd">HYD:</span> '+p.hydrophobic.join(', '):''}</div>`;
        card.addEventListener('click',()=>focusPocket(p.id));
        list.appendChild(card);
    });
}

function focusPocket(id){
    const p=POCKETS.find(x=>x.id===id); if(!p) return;
    document.querySelectorAll('.pocket-card').forEach(c=>c.classList.remove('active'));
    const card=document.querySelector(`[data-pocket-id="${id}"]`);
    if(card) card.classList.add('active');
    if(p.lining.length) viewer.zoomTo({chain:p.chain||'A',resi:p.lining},800);
    const info=document.getElementById('pocket-info');
    info.style.display='block';
    document.getElementById('info-name').textContent=`Site ${p.id} — ${p.type}`;
    document.getElementById('info-detail').innerHTML=`Quality: ${p.quality_score.toFixed(1)} · Druggability: ${p.druggability_score.toFixed(2)}<br>${p.lining.length} lining · ${p.catalytic.length} catalytic · ${p.aromatic.length} aromatic · ${p.hydrophobic.length} hydrophobic`;
    activePocket=id; viewer.render();
}

function toggleProtein(btn){
    showProtein=!showProtein; btn.classList.toggle('active');
    if(showProtein){
        viewer.setStyle({},{cartoon:{color:'gray',opacity:0.7}});
        POCKETS.forEach(p=>{const c=p.chain||'A';
            if(p.lining.length) viewer.setStyle({chain:c,resi:p.lining},{stick:{radius:0.12,color:'gray'},cartoon:{color:'gray',opacity:0.7}});
            if(p.catalytic.length) viewer.setStyle({chain:c,resi:p.catalytic},{stick:{radius:0.15,color:0xff3366},cartoon:{color:'gray',opacity:0.7}});
            if(p.aromatic.length) viewer.setStyle({chain:c,resi:p.aromatic},{stick:{radius:0.15,color:0x00ff88},cartoon:{color:'gray',opacity:0.7}});
            if(p.hydrophobic.length) viewer.setStyle({chain:c,resi:p.hydrophobic},{stick:{radius:0.15,color:0xffaa00},cartoon:{color:'gray',opacity:0.7}});
        });
    } else {
        viewer.setStyle({},{cartoon:{hidden:true}});
        POCKETS.forEach(p=>{const c=p.chain||'A';
            if(p.lining.length) viewer.setStyle({chain:c,resi:p.lining},{stick:{radius:0.12,color:'gray'}});
            if(p.catalytic.length) viewer.setStyle({chain:c,resi:p.catalytic},{stick:{radius:0.15,color:0xff3366}});
            if(p.aromatic.length) viewer.setStyle({chain:c,resi:p.aromatic},{stick:{radius:0.15,color:0x00ff88}});
            if(p.hydrophobic.length) viewer.setStyle({chain:c,resi:p.hydrophobic},{stick:{radius:0.15,color:0xffaa00}});
        });
    }
    viewer.render();
}

function toggleSurfaces(btn){
    showSurfaces=!showSurfaces; btn.classList.toggle('active');
    Object.values(pocketSurfaces).forEach(sp=>{
        if(sp&&sp.then) sp.then(s=>{viewer.setSurfaceMaterialStyle(s.surfid!==undefined?s.surfid:s,{opacity:showSurfaces?0.25:0.0});viewer.render();});
    }); viewer.render();
}

function toggleCentroids(btn){
    showCentroids=!showCentroids; btn.classList.toggle('active');
    if(showCentroids){
        POCKETS.forEach(p=>{if(p.centroid){centroidSpheres.push(viewer.addSphere({center:{x:p.centroid[0],y:p.centroid[1],z:p.centroid[2]},radius:1.2,color:TYPE_COLORS[p.type]||0xffaa00,opacity:0.8}));}});
    } else { centroidSpheres.forEach(s=>viewer.removeShape(s)); centroidSpheres=[]; }
    viewer.render();
}

function toggleLabels(btn){
    showLabels=!showLabels; btn.classList.toggle('active');
    if(showLabels){
        POCKETS.forEach(p=>{if(p.centroid){
            const col=TYPE_COLORS[p.type]?'#'+TYPE_COLORS[p.type].toString(16).padStart(6,'0'):'#ffaa00';
            labelSprites.push(viewer.addLabel(`S${p.id} ${p.type}`,{position:{x:p.centroid[0],y:p.centroid[1],z:p.centroid[2]},fontSize:12,fontColor:'white',backgroundColor:'rgba(0,0,0,0.7)',borderColor:col,borderThickness:1,padding:4}));
        }});
    } else { labelSprites.forEach(l=>viewer.removeLabel(l)); labelSprites=[]; }
    viewer.render();
}

function showAll(){POCKETS.forEach(p=>{const c=p.chain||'A';if(p.lining.length) viewer.setStyle({chain:c,resi:p.lining},{stick:{radius:0.12,color:'gray'},cartoon:{color:'gray',opacity:0.7}});}); viewer.render();}
function hideAll(){viewer.setStyle({},{cartoon:{color:'gray',opacity:0.7}}); viewer.render();}
function resetView(){viewer.zoomTo();document.getElementById('pocket-info').style.display='none';document.querySelectorAll('.pocket-card').forEach(c=>c.classList.remove('active'));activePocket=null;viewer.render();}

document.addEventListener('DOMContentLoaded',initViewer);
</script>
</body>
</html>'''
    return html

def main():
    if len(sys.argv) < 2:
        print("Usage: generate_viewer.py <output_dir> [source_pdb]"); sys.exit(1)
    output_dir = Path(sys.argv[1])
    dirname = output_dir.name
    pdb_id = dirname.split('_')[0] if '_' in dirname else dirname
    pml_files = list(output_dir.glob('*.binding_sites.pml'))
    bs_pdb_files = list(output_dir.glob('*.binding_sites.pdb'))
    if not pml_files: print(f"ERROR: No .binding_sites.pml found in {output_dir}"); sys.exit(1)
    if not bs_pdb_files: print(f"ERROR: No .binding_sites.pdb found in {output_dir}"); sys.exit(1)
    source_pdb = None
    if len(sys.argv) > 2: source_pdb = sys.argv[2]
    else:
        candidate = Path(f'/opt/prism4d/samples/{pdb_id}.pdb')
        if candidate.exists(): source_pdb = str(candidate)
    if not source_pdb or not Path(source_pdb).exists():
        print(f"ERROR: Cannot find source PDB for {pdb_id}"); sys.exit(1)
    print(f"  PML:        {pml_files[0]}")
    print(f"  Sites PDB:  {bs_pdb_files[0]}")
    print(f"  Source PDB: {source_pdb}")
    pockets = parse_pml(str(pml_files[0]))
    scores = parse_binding_sites_pdb(str(bs_pdb_files[0]))
    with open(source_pdb) as f: pdb_content = f.read()
    print(f"  Pockets:    {len(pockets)}")
    html = generate_html(pockets, scores, pdb_content, pdb_id)
    out_html = output_dir / f'{pdb_id}_viewer.html'
    with open(out_html, 'w') as f: f.write(html)
    print(f"  Viewer:     {out_html}")

if __name__ == '__main__':
    main()
