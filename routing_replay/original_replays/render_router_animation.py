from pathlib import Path
import json, math, shutil, zipfile
from PIL import Image, ImageDraw, ImageFont

OUT=Path('C:/Users/Admin/Documents/Codex/2026-11-04/ca/outputs/router_animation')
data=json.loads((OUT/'trace.json').read_text())
W,H=1200,800
BG='#101C25';FG='#ECF3F4';MUTED='#9BB2BE';GRID='#253642';TEAL='#42D5BE';AMBER='#EDAD62';BLUE='#5ABBE2'
fontfile='C:/Windows/Fonts/segoeui.ttf';boldfile='C:/Windows/Fonts/segoeuib.ttf'
def font(n,b=False):return ImageFont.truetype(boldfile if b else fontfile,n)
def text(d,xy,s,size=20,fill=FG,bold=False):d.text(xy,s,font=font(size,bold),fill=fill)
def base(title,subtitle):
 im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
 text(d,(48,25),'QWEN3.5-4B  /  ACTUAL ROUTER REPLAY',17,MUTED,True)
 text(d,(48,58),title,35,FG,True);text(d,(48,109),subtitle,19,MUTED)
 d.line((48,142,1152,142),fill=GRID,width=1)
 text(d,(48,744),'Saved dense Q/K • layer 3, head 0 (zero-based) • C4/L6/R16 • p=.90',17,MUTED)
 text(d,(48,770),'Diagnostic replay, not the 96-prompt benchmark. Animation timing is illustrative.',16,MUTED)
 return im,d

row=data['states']['trained'][-1];end=row['end'];levels=row['levels'];x0,x1=190,960
def xx(pos):return x0+(x1-x0)*pos/end
frames=[]
for frame in range(144):
 stage=min(6,frame//20);within=(frame%20)/20
 im,d=base('How one query searches the tree',f'Trained router • query position {end-1:,} • {end:,} visible keys')
 for lx,color,label in [(190,AMBER,'scoring'),(345,TEAL,'retained'),(505,'#6A4A31','rejected')]:
  d.rectangle((lx,160,lx+12,172),fill=color);text(d,(lx+20,154),label,15,MUTED)
 text(d,(700,154),'Incomplete-parent groups join lower levels',14,MUTED)
 for idx,level in enumerate(levels):
  y=198+idx*68;span=level['span'];active=idx==stage;done=idx<stage
  text(d,(48,y-5),f"Level {level['level']}",20,FG,True)
  text(d,(48,y+20),f'{span:,} keys / node',14,MUTED)
  d.rectangle((x0,y,x1,y+27),fill=GRID)
  if active or done:
   selected=set(level['selected']);show_selected=done or within>=.45
   for node in level['candidates']:
    left,right=xx(1+node*span),xx(1+(node+1)*span)
    color=TEAL if show_selected and node in selected else ('#6A4A31' if show_selected else AMBER)
    d.rectangle((left,y,max(left+1,right-1),y+27),fill=color)
   count=len(level['selected']) if show_selected else len(level['candidates'])
   label=f"{count} kept / {len(level['candidates'])} scored" if show_selected else f'{count} candidates'
   text(d,(978,y+2),label,15,TEAL if show_selected else AMBER)
  else:text(d,(978,y+2),'not reached yet',15,MUTED)
  if idx<5 and done:
   # Spatial interval connections, limited to coarse levels for legibility.
   if idx<2:
    for node in level['selected']:
     center=xx(1+(node+.5)*span)
     for child in levels[idx+1]['candidates']:
      if child//4==node:
       d.line((center,y+28,xx(1+(child+.5)*(span//4)),y+68),fill='#31585E',width=1)
 y=638
 text(d,(48,y-4),'Original keys',19,FG,True)
 d.rectangle((x0,y,x1,y+27),fill=GRID)
 if stage==6:
  for key in row['selected_keys']:d.line((xx(key),y,xx(key),y+27),fill=TEAL,width=1)
  text(d,(978,y+2),f"{len(row['selected_keys'])} selected",15,TEAL)
  caption=f"Attend over {len(row['selected_keys']):,} original keys • sink and current chunk included"
 else:
  lev=levels[stage];action='Keep the top-p set; expand its children' if within>=.45 else 'Score candidate summaries for this query'
  caption=f"{action}  |  level {lev['level']}"
 text(d,(190,689),caption,19,FG)
 d.rectangle((48,724,48+1104*(frame+1)/144,727),fill=TEAL)
 frames.append(im)
frames[130].save(OUT/'tree_traversal_poster.png')
frames[0].save(OUT/'tree_traversal.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
del frames

frames=[]
for frame in range(128):
 j=frame//16;im,d=base('Different queries select different keys',
  f'Same saved query and keys for both routers • snapshot {j+1}/8 • not consecutive generation steps')
 for col,mode in enumerate(('untrained','trained')):
  r=data['states'][mode][j];color=AMBER if col==0 else TEAL;x=70+col*590;y=211
  text(d,(x,160),'Original initialization' if col==0 else 'Trained references',25,color,True)
  mask=Image.new('RGB',(128,128),GRID);pix=mask.load()
  for k in range(r['end'],16384):pix[k%128,k//128]=(16,28,37)
  for k in r['selected_keys']:pix[k%128,k//128]=tuple(bytes.fromhex(color[1:]))
  mask=mask.resize((384,384),Image.Resampling.NEAREST);im.paste(mask,(x,y))
  text(d,(x,y+393),'1 cell = 1 key position • left to right, then down',15,MUTED)
  text(d,(x,638),f"{len(r['selected_keys']):,} / {r['end']:,} keys",25,color,True)
  text(d,(x,676),f"True dense mass retained: {r['dense_mass_retained']*100:.1f}%",18,FG)
  d.rectangle((x,715,x+384*(j+1)/8,718),fill=color)
 frames.append(im)
frames[112].save(OUT/'query_replay_poster.png')
frames[0].save(OUT/'query_replay.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
notes='''ACTUAL ROUTER REPLAYS — CAPTIONS AND LIMITS

tree_traversal.gif: A trained C4/L6/R16 router descends six summary levels for one saved Qwen query, then selects original key positions. Amber is a candidate being scored; teal is retained; subdued amber is rejected at that level. Counts are actual decisions from the existing CPU reference implementation, not invented illustration. The sink and current chunk are added outside the tree.

query_replay.gif: Original initialization and trained references receive identical saved dense-model queries and keys. Eight saved query positions are shown in order; they are not consecutive autoregressive steps. Every colored cell is an actual selected key. Cells are laid out left-to-right then top-to-bottom. Uncolored visible keys remain stored but are not selected. True dense mass is computed on the same query/key pair for each snapshot. It is not an answer-quality score.

Provenance: layer 3 and query head 0, both zero-based; final eight query positions in doc0_layer3.pt from the earlier plan_large teacher cache. Positions were chosen by fixed index, not ranked by outcomes. This is a diagnostic sample, not a fresh held-out benchmark or a final 96-prompt test trace. No generated answer or semantic importance is asserted. Both GIFs loop with illustrative timing; neither measures performance or demonstrates a speedup. All 16 selected-key sets were checked against the existing selected_indices reference.

Source tensor/checkpoint hashes, full selection traces, counts and exact positions are recorded in trace.json. capture_router_animation.py documents the capture, using the original local environment. render_router_animation.py can render trace.json using Pillow (change OUT and font paths if needed). No base-model weights, teacher tensor contents or original corpus text are redistributed.

Suggested caption: A replay of my Qwen3.5-4B hierarchical router on saved model queries and keys. It searches coarse summaries, descends selected branches, and gathers original keys. This shows actual reference-router decisions; animation timing is illustrative, and this is not a speed demo.
'''
(OUT/'README_ANIMATIONS.txt').write_text(notes,encoding='utf8')
for name in ['capture_router_animation.py','render_router_animation.py']:
 src=Path(__file__).parent/name
 if src.resolve()!=(OUT/name).resolve():shutil.copy2(src,OUT/name)
with zipfile.ZipFile(OUT.parent/'qwen_router_animations.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.iterdir():
  if p.is_file():z.write(p,p.name)
print([(p.name,p.stat().st_size) for p in OUT.glob('*.gif')])
