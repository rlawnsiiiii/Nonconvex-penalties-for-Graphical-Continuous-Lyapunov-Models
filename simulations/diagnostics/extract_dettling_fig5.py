"""Extract exact plotted values + error bars from Dettling Figure 5 (vector PDF)."""
import pymupdf, json, sys, numpy as np
pdf = sys.argv[1]
pg = pymupdf.open(pdf)[10]
SERIES = {(0.0,0.0,1.0):"C_ID", (1.0,0.0,0.0):"C_Random_Diag",
          (0.0,1.0,0.0):"C_Random_Min_Diag", (0.627,0.125,0.941):"C_Random_Full"}
col = lambda d: tuple(round(x,3) for x in (d.get("fill") or d.get("color") or (9,9,9)))
draws = pg.get_drawings()

# panels: (metric, x-range of plotting area, y-range of plotting area)
words = [w for w in pg.get_text("words") if w[4].replace('.','',1).isdigit()]
PANELS = {"max_acc":(150,305,105,210), "max_f1":(333,490,105,210),
          "auc":(150,305,230,335), "aupr":(333,490,230,335)}
def panel_of(x, y):
    for m,(x0,x1,y0,y1) in PANELS.items():
        if x0 <= x <= x1 and y0 <= y <= y1: return m
    return None

# y-maps from tick labels, snapped to the nearest horizontal gridline
grid = [d for d in draws if col(d)==(0.922,0.922,0.922)]
hlines = [((d["rect"].y0+d["rect"].y1)/2, d["rect"].x0, d["rect"].x1) for d in grid
          if abs(d["rect"].y1-d["rect"].y0) < 0.01]
ymap = {}
for m,(x0,x1,y0,y1) in PANELS.items():
    labs = [w for w in words if (x0-30) <= (w[0]+w[2])/2 <= x0 and y0 <= (w[1]+w[3])/2 <= y1+10
            and '.' in w[4]]
    pts = []
    for w in labs:
        yc = (w[1]+w[3])/2
        cand = [h for h in hlines if h[1] <= x0+5 and h[2] >= x0+30]
        yg = min(cand, key=lambda h: abs(h[0]-yc))[0] if cand else yc
        pts.append((yg, float(w[4])))
    (a,b) = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
    ymap[m] = (a,b, pts)
# x-map: p tick labels (same for every panel column)
def xmap_for(m):
    x0,x1,y0,y1 = PANELS[m]
    labs=[w for w in words if x0-10 <= (w[0]+w[2])/2 <= x1+10 and y1 <= (w[1]+w[3])/2 <= y1+15
          and '.' not in w[4]]
    return np.polyfit([(w[0]+w[2])/2 for w in labs], [float(w[4]) for w in labs], 1)

out = {}
for rgb, name in SERIES.items():
    mine = [d for d in draws if col(d)==rgb]
    marks = [d for d in mine if d["type"]=="fs" and len(d["items"])==4]
    segs  = [d for d in mine if d["type"]=="s" and len(d["items"])==1]
    vert  = [d for d in segs if abs(d["rect"].x1-d["rect"].x0) < 0.01]
    for d in marks:
        cx=(d["rect"].x0+d["rect"].x1)/2; cy=(d["rect"].y0+d["rect"].y1)/2
        m = panel_of(cx, cy)
        if m is None: continue                                   # legend
        a,b,_ = ymap[m]; xa,xb = xmap_for(m)
        p = int(round(xa*cx+xb)); val = a*cy+b
        vb = [v for v in vert if abs(v["rect"].x0-cx) < 0.3 and panel_of(cx,(v["rect"].y0+v["rect"].y1)/2)==m]
        se = None
        if vb:
            v = min(vb, key=lambda v: abs((v["rect"].y0+v["rect"].y1)/2-cy))
            se = abs(a)*(v["rect"].y1-v["rect"].y0)/2
        out.setdefault(m,{}).setdefault(name,{})[p] = {"mean":round(val,4),"se":round(se,4) if se else None}
json.dump(out, open(sys.argv[2],"w"), indent=1)
for m in ("max_acc","max_f1","auc","aupr"):
    print(f"[{m}]  tick fit residual: {max(abs(ymap[m][0]*y+ymap[m][1]-v) for y,v in ymap[m][2]):.1e}")
    for name in SERIES.values():
        row = out[m][name]
        print(f"  {name:<18}" + "  ".join(f"p{p}:{row[p]['mean']:.3f}±{row[p]['se']:.3f}" for p in sorted(row)))
