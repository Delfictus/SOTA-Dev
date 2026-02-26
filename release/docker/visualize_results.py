#!/usr/bin/env python3
"""
PRISM-4D Demo — Interactive 3D Binding Site Visualizer

Generates a self-contained HTML file with 3Dmol.js that displays:
  - Protein structure (ribbon + surface)
  - Detected binding site pocket (colored mesh)
  - Lining residues (labeled sticks)
  - Pharmacophore features (colored spheres)

Usage:
  python3 visualize_results.py <binding_sites.json> [--pdb <pdb_id>]
  python3 visualize_results.py results/2fhz/2fhz.binding_sites.json

Output:
  results/<target>/<target>_viewer.html  (open in any browser)
"""

import json
import sys
import os
import argparse
from pathlib import Path

def generate_viewer_html(binding_data, pdb_id, output_path, topology_path=None):
    """Generate a self-contained 3Dmol.js HTML viewer for binding site results."""

    sites = binding_data.get("sites", [])
    if not sites:
        # Fallback: binding_sites might be the list in some formats
        bs = binding_data.get("binding_sites", [])
        if isinstance(bs, list):
            sites = bs
    if not sites:
        print("ERROR: No binding sites found in JSON")
        return False

    target = binding_data.get("structure", binding_data.get("target", pdb_id))
    n_sites = len(sites)

    # Build site data for JS
    site_js_data = []
    for i, site in enumerate(sites):
        centroid = site.get("centroid", site.get("center", [0, 0, 0]))
        volume = site.get("volume", 0)
        quality = site.get("quality_score", site.get("quality", 0))
        drug = site.get("druggability", site.get("druggability_score", 0))
        residues = site.get("lining_residues", site.get("residues", []))
        classification = site.get("classification", "Unknown")

        # Parse residues
        res_list = []
        for r in residues:
            if isinstance(r, dict):
                res_list.append({
                    "chain": r.get("chain", "A"),
                    "resi": r.get("resid", r.get("residue_id", r.get("id", 0))),
                    "name": r.get("resname", r.get("residue_name", r.get("name", "UNK"))),
                    "dist": r.get("min_distance", r.get("distance", 0)),
                    "catalytic": r.get("is_catalytic", False)
                })
            elif isinstance(r, str):
                # Parse "A:LEU55" format
                parts = r.replace(":", " ").split()
                if len(parts) >= 2:
                    chain = parts[0]
                    name = ''.join(c for c in parts[1] if c.isalpha())
                    resi = ''.join(c for c in parts[1] if c.isdigit())
                    res_list.append({"chain": chain, "resi": int(resi) if resi else 0, "name": name, "dist": 0})

        site_js_data.append({
            "index": i,
            "centroid": centroid,
            "volume": volume,
            "quality": quality,
            "druggability": drug,
            "classification": classification,
            "residues": res_list
        })

    sites_json = json.dumps(site_js_data)

    # Determine PDB source
    pdb_id_clean = pdb_id.lower().strip() if pdb_id else target.lower().strip()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PRISM-4D | {target.upper()} Binding Site Viewer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
<style>
:root{{--bg:#06060a;--card:#0f0f1a;--cyan:#00e5ff;--green:#00ff88;--red:#ff3366;--amber:#ffaa00;--purple:#aa44ff;--text:#e8ecf0;--dim:#556070;--border:rgba(0,229,255,0.12);--mono:'Courier New',monospace}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',sans-serif;overflow:hidden}}
.header{{background:var(--cyan);color:var(--bg);text-align:center;padding:4px;font-family:var(--mono);font-size:11px;font-weight:700;letter-spacing:3px}}
.container{{display:grid;grid-template-columns:1fr 320px;height:calc(100vh - 28px)}}
#viewer{{width:100%;height:100%;background:var(--card)}}
.panel{{background:var(--card);border-left:1px solid var(--border);overflow-y:auto;padding:16px}}
.panel h2{{font-size:16px;color:var(--cyan);margin-bottom:12px;font-family:var(--mono);letter-spacing:1px}}
.panel h3{{font-size:13px;color:var(--text);margin:16px 0 8px;padding-top:12px;border-top:1px solid var(--border)}}
.site-btn{{display:block;width:100%;padding:10px 12px;margin-bottom:6px;background:rgba(0,229,255,0.05);border:1px solid var(--border);border-radius:4px;color:var(--text);font-family:var(--mono);font-size:12px;cursor:pointer;text-align:left;transition:0.2s}}
.site-btn:hover,.site-btn.active{{border-color:var(--cyan);background:rgba(0,229,255,0.12);color:var(--cyan)}}
.kv{{display:flex;justify-content:space-between;padding:4px 0;font-size:12px;border-bottom:1px solid rgba(255,255,255,0.03)}}
.kv-k{{color:#8892a0}}.kv-v{{color:var(--text);font-family:var(--mono);font-size:11px}}
.ctrl-row{{display:flex;gap:4px;flex-wrap:wrap;margin:8px 0}}
.ctrl-btn{{flex:1;min-width:60px;padding:6px 4px;background:rgba(0,229,255,0.05);border:1px solid var(--border);border-radius:3px;color:#8892a0;font-family:var(--mono);font-size:10px;cursor:pointer;text-align:center;transition:0.2s;text-transform:uppercase;letter-spacing:1px}}
.ctrl-btn:hover,.ctrl-btn.on{{border-color:var(--cyan);color:var(--cyan);background:rgba(0,229,255,0.12)}}
.res-list{{max-height:200px;overflow-y:auto;font-family:var(--mono);font-size:11px;line-height:1.8}}
.res-item{{padding:2px 6px;border-radius:2px;cursor:pointer;transition:0.15s}}
.res-item:hover{{background:rgba(0,229,255,0.08);color:var(--cyan)}}
.cat{{color:var(--red);font-weight:700}}
.quality-bar{{height:6px;border-radius:3px;background:var(--border);margin-top:4px;overflow:hidden}}
.quality-fill{{height:100%;border-radius:3px;transition:width 0.5s}}
.legend{{font-size:10px;color:var(--dim);margin-top:16px;padding-top:12px;border-top:1px solid var(--border);line-height:1.8}}
.legend span{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}}
</style>
</head>
<body>
<div class="header">PRISM-4D &mdash; {target.upper()} CRYPTIC BINDING SITE ANALYSIS</div>
<div class="container">
<div id="viewer"></div>
<div class="panel">
<h2>{target.upper()}</h2>
<div class="kv"><span class="kv-k">PDB</span><span class="kv-v">{pdb_id_clean.upper()}</span></div>
<div class="kv"><span class="kv-k">Sites Found</span><span class="kv-v">{n_sites}</span></div>
<div class="kv"><span class="kv-k">Engine</span><span class="kv-v">PRISM-4D nhs_rt_full</span></div>

<h3>Detected Sites</h3>
<div id="site-buttons"></div>

<h3>Display</h3>
<div class="ctrl-row">
  <button class="ctrl-btn on" onclick="toggle('ribbon')">Ribbon</button>
  <button class="ctrl-btn" onclick="toggle('surface')">Surface</button>
  <button class="ctrl-btn on" onclick="toggle('pocket')">Pocket</button>
  <button class="ctrl-btn on" onclick="toggle('residues')">Residues</button>
  <button class="ctrl-btn" onclick="toggle('labels')">Labels</button>
  <button class="ctrl-btn on" onclick="toggleSpin()">Spin</button>
</div>

<h3>Site Details</h3>
<div id="site-details"></div>

<h3>Lining Residues</h3>
<div id="res-list" class="res-list"></div>

<div class="legend">
  <span style="background:var(--cyan)"></span> Pocket centroid &nbsp;
  <span style="background:var(--green)"></span> Hydrophobic &nbsp;
  <span style="background:var(--red)"></span> Catalytic &nbsp;
  <span style="background:var(--amber)"></span> Polar &nbsp;
  <span style="background:var(--purple)"></span> Aromatic
</div>

<div style="margin-top:16px;font-size:10px;color:var(--dim);font-family:var(--mono)">
  Generated by PRISM-4D<br>
  &copy; Delfictus IO LLC
</div>
</div>
</div>

<script>
var sites = {sites_json};
var pdbId = '{pdb_id_clean}';
var viewer = null;
var pdbData = null;
var currentSite = 0;
var layers = {{ribbon:true, surface:false, pocket:true, residues:true, labels:false}};
var spinning = true;

// Color palette for multi-site
var siteColors = ['0x00e5ff','0x00ff88','0xffaa00','0xff3366','0xaa44ff','0xff8844'];

function init() {{
  var el = document.getElementById('viewer');
  el.innerHTML = '<div style="color:#00e5ff;text-align:center;padding-top:40vh;font-family:monospace">Loading ' + pdbId.toUpperCase() + ' from RCSB...</div>';

  fetch('https://files.rcsb.org/download/' + pdbId.toUpperCase() + '.pdb')
    .then(function(r) {{ return r.text(); }})
    .then(function(data) {{
      pdbData = data;
      el.innerHTML = '';
      viewer = $3Dmol.createViewer(el, {{backgroundColor:'#0f0f1a', antialias:true}});
      viewer.addModel(pdbData, 'pdb');
      buildSiteButtons();
      renderScene();
      selectSite(0);
      if(spinning) viewer.spin('y', 0.4);
      new ResizeObserver(function(){{ if(viewer){{viewer.resize();viewer.render();}} }}).observe(el);
    }})
    .catch(function() {{
      el.innerHTML = '<div style="color:#ff3366;text-align:center;padding-top:40vh;font-family:monospace">Failed to load PDB. Check network.</div>';
    }});
}}

function buildSiteButtons() {{
  var html = '';
  sites.forEach(function(s, i) {{
    var color = siteColors[i % siteColors.length];
    var q = (s.quality * 100).toFixed(0);
    html += '<button class="site-btn' + (i===0?' active':'') + '" onclick="selectSite(' + i + ')" id="sbtn-' + i + '">';
    html += 'Site ' + i + ' &mdash; ' + s.classification + '<br>';
    html += '<span style="font-size:10px;color:#8892a0">Vol: ' + s.volume.toFixed(0) + 'A\\u00B3 | Q: ' + q + '% | Drug: ' + (s.druggability*100).toFixed(0) + '%</span>';
    html += '</button>';
  }});
  document.getElementById('site-buttons').innerHTML = html;
}}

function selectSite(idx) {{
  currentSite = idx;
  // Update buttons
  document.querySelectorAll('.site-btn').forEach(function(b,i){{
    b.className = 'site-btn' + (i===idx?' active':'');
  }});
  // Update details
  var s = sites[idx];
  var html = '';
  html += '<div class="kv"><span class="kv-k">Classification</span><span class="kv-v">' + s.classification + '</span></div>';
  html += '<div class="kv"><span class="kv-k">Centroid</span><span class="kv-v">(' + s.centroid.map(function(v){{return v.toFixed(1);}}).join(', ') + ')</span></div>';
  html += '<div class="kv"><span class="kv-k">Volume</span><span class="kv-v">' + s.volume.toFixed(0) + ' A\\u00B3</span></div>';
  html += '<div class="kv"><span class="kv-k">Quality</span><span class="kv-v">' + (s.quality*100).toFixed(1) + '%</span></div>';
  html += '<div class="quality-bar"><div class="quality-fill" style="width:' + (s.quality*100) + '%;background:' + (s.quality>0.6?'var(--green)':s.quality>0.3?'var(--amber)':'var(--red)') + '"></div></div>';
  html += '<div class="kv" style="margin-top:8px"><span class="kv-k">Druggability</span><span class="kv-v">' + (s.druggability*100).toFixed(1) + '%</span></div>';
  html += '<div class="quality-bar"><div class="quality-fill" style="width:' + (s.druggability*100) + '%;background:' + (s.druggability>0.5?'var(--green)':s.druggability>0.25?'var(--amber)':'var(--red)') + '"></div></div>';
  html += '<div class="kv" style="margin-top:8px"><span class="kv-k">Residues</span><span class="kv-v">' + s.residues.length + '</span></div>';
  document.getElementById('site-details').innerHTML = html;

  // Update residue list
  var rhtml = '';
  s.residues.forEach(function(r) {{
    rhtml += '<div class="res-item" onclick="zoomResidue(\\'' + r.chain + '\\',' + r.resi + ')">';
    rhtml += r.chain + ':' + r.name + r.resi;
    if(r.dist > 0) rhtml += ' <span style="color:var(--dim)">' + r.dist.toFixed(1) + 'A</span>';
    rhtml += '</div>';
  }});
  document.getElementById('res-list').innerHTML = rhtml;

  renderScene();
  // Zoom to site
  var c = s.centroid;
  viewer.zoomTo({{center:{{x:c[0],y:c[1],z:c[2]}}, spec:{{within:{{distance:12,sel:{{}}}} }} }});
  viewer.render();
}}

function renderScene() {{
  if(!viewer) return;
  viewer.removeAllShapes();
  viewer.removeAllSurfaces();
  viewer.removeAllLabels();
  viewer.setStyle({{}}, {{}}); // clear

  var s = sites[currentSite];
  var color = siteColors[currentSite % siteColors.length];

  // Ribbon
  if(layers.ribbon) {{
    viewer.setStyle({{}}, {{cartoon:{{color:'spectrum', opacity:0.85, thickness:0.8}}}});
  }}

  // Surface
  if(layers.surface) {{
    viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity:0.15, color:'#00e5ff'}}, {{}});
  }}

  // Pocket sphere
  if(layers.pocket) {{
    var c = s.centroid;
    viewer.addSphere({{center:{{x:c[0],y:c[1],z:c[2]}}, radius:2.0, color:color, opacity:0.7}});
    // Pocket boundary
    viewer.addSphere({{center:{{x:c[0],y:c[1],z:c[2]}}, radius:8.0, color:color, opacity:0.08, wireframe:true}});
  }}

  // Lining residues
  if(layers.residues && s.residues.length > 0) {{
    s.residues.forEach(function(r) {{
      viewer.setStyle({{chain:r.chain, resi:r.resi}}, {{
        stick:{{radius:0.15, color:'#00e5ff'}},
        cartoon:{{color:'spectrum', opacity:0.85, thickness:0.8}}
      }});
    }});
  }}

  // Labels
  if(layers.labels && s.residues.length > 0) {{
    s.residues.forEach(function(r) {{
      viewer.addLabel(r.chain + ':' + r.name + r.resi, {{
        position:{{resi:r.resi, chain:r.chain}},
        font:'Arial', fontSize:11, fontColor:'#ffffff',
        backgroundColor:'#000000', backgroundOpacity:0.6,
        showBackground:true
      }}, {{chain:r.chain, resi:r.resi, atom:'CA'}});
    }});
  }}

  viewer.render();
}}

function toggle(layer) {{
  layers[layer] = !layers[layer];
  // Update button
  document.querySelectorAll('.ctrl-btn').forEach(function(b) {{
    if(b.textContent.trim().toLowerCase() === layer.toLowerCase()) {{
      b.className = 'ctrl-btn' + (layers[layer]?' on':'');
    }}
  }});
  renderScene();
}}

function toggleSpin() {{
  spinning = !spinning;
  if(spinning) viewer.spin('y', 0.4); else viewer.spin(false);
  document.querySelectorAll('.ctrl-btn').forEach(function(b) {{
    if(b.textContent.trim().toLowerCase() === 'spin') {{
      b.className = 'ctrl-btn' + (spinning?' on':'');
    }}
  }});
}}

function zoomResidue(chain, resi) {{
  viewer.zoomTo({{chain:chain, resi:resi}});
  viewer.render();
}}

init();
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    return True


def main():
    parser = argparse.ArgumentParser(description='PRISM-4D Binding Site 3D Viewer Generator')
    parser.add_argument('json_file', help='Path to binding_sites.json')
    parser.add_argument('--pdb', help='PDB ID (auto-detected from filename if omitted)')
    parser.add_argument('--output', '-o', help='Output HTML path (default: same dir as input)')
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"ERROR: File not found: {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    # Auto-detect PDB ID from filename
    pdb_id = args.pdb
    if not pdb_id:
        stem = json_path.stem.replace('.binding_sites', '').replace('_binding_sites', '')
        pdb_id = stem

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = json_path.parent / f"{pdb_id}_viewer.html"

    print(f"PRISM-4D 3D Viewer Generator")
    print(f"  Input:  {json_path}")
    print(f"  PDB:    {pdb_id.upper()}")
    print(f"  Output: {out_path}")
    print()

    if generate_viewer_html(data, pdb_id, str(out_path)):
        size_kb = out_path.stat().st_size / 1024
        print(f"  Viewer generated: {out_path} ({size_kb:.1f} KB)")
        print(f"  Open in browser:  file://{out_path.resolve()}")
        print()
        print(f"  The viewer fetches the PDB structure from RCSB at load time.")
        print(f"  All PRISM-4D binding site data is embedded in the HTML.")
    else:
        print("ERROR: Failed to generate viewer")
        sys.exit(1)


if __name__ == '__main__':
    main()
