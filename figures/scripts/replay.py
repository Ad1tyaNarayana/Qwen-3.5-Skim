import json,sys
from PIL import Image,ImageDraw,ImageFont
t=json.load(open(sys.argv[1]))
S=2;BG=(10,10,10);OR=(255,107,26);ORD=(110,52,18);TX=(242,239,233);MU=(125,122,117);DIM=(40,40,40);WH=(250,250,250);CELL=(26,26,26)
F='/usr/share/fonts/opentype/inter/Inter-%s.otf'
def f(w,s):return ImageFont.truetype(F%w,s*S)
FOOT='Routing demonstration on saved Q/K from an earlier WikiText training-window capture · layer 3, head 0 · C4/L6/R16 · p = .90. Not benchmark evidence; timing illustrative.'
def header(d,title,sub):
    d.text((48*S,34*S),'QWEN3.5-4B  /  ZHA-STYLE ROUTER  /  ACTUAL ROUTER REPLAY',font=f('Bold',13),fill=OR)
    d.text((48*S,56*S),title,font=f('Bold',30),fill=TX)
    d.text((48*S,98*S),sub,font=f('Regular',15),fill=MU)
# ---------- A: tree traversal, trained, last snapshot ----------
W,H=1200,760
s=t['states']['trained'][-1];end=s['end'];lv=s['levels']
X0,X1=200,1000;sc=lambda k:X0+(X1-X0)*k/end
rowy=lambda r:170+r*70
def tree_frame(stage,phase):
    im=Image.new('RGB',(W*S,H*S),BG);d=ImageDraw.Draw(im)
    header(d,'How one query searches the tree',f'Trained router · query position {end-1:,} · {end:,} visible keys · real candidate and kept nodes, drawn to scale')
    for r,L in enumerate(lv):
        y=rowy(r);span=L['span'];lev=L['level']
        d.text((48*S,(y-11)*S),f'Level {lev}',font=f('SemiBold',15),fill=TX if r<=stage else MU)
        d.text((48*S,(y+8)*S),f'{span:,} keys / node',font=f('Regular',11),fill=MU)
        d.rectangle([X0*S,(y-6)*S,X1*S,(y+6)*S],fill=CELL)
        if r>stage:
            d.text((1020*S,(y-8)*S),'not reached',font=f('Regular',13),fill=(70,70,70));continue
        sel=set(L['selected']);done=r<stage or phase==1
        for c in L['candidates']:
            a=1+c*span;b=a+span
            col=(OR if c in sel else ORD) if done else WH
            x0=sc(a);x1=max(sc(b)-1.2,x0+1.6)
            d.rectangle([x0*S,(y-6)*S,x1*S,(y+6)*S],fill=col)
        txt=f"{len(L['candidates'])} scored → {len(sel)} kept" if done else f"scoring {len(L['candidates'])}"
        d.text((1020*S,(y-8)*S),txt,font=f('SemiBold' if r==stage else 'Regular',13),fill=OR if done else WH)
    y=rowy(6)+20
    d.text((48*S,(y-9)*S),'Original keys',font=f('SemiBold',15),fill=TX)
    d.rectangle([X0*S,(y-8)*S,X1*S,(y+8)*S],fill=CELL)
    if stage>=6:
        for k in s['selected_keys']:
            x=sc(k);d.rectangle([x*S,(y-8)*S,max(sc(k+1),x+1.6)*S,(y+8)*S],fill=OR)
        d.text((1020*S,(y-9)*S),f"{len(s['selected_keys'])} of {end:,} keys",font=f('SemiBold',13),fill=OR)
        msg=f"Attend to {len(s['selected_keys'])} keys (incl. sink + current chunk) · {s['dense_mass_retained']*100:.1f}% of true dense attention mass retained"
    else:
        msg=f"Level {lv[stage]['level']}: "+('keep the smallest set covering p = .90 of estimated mass; descend only into those' if phase else 'score candidate summaries against the query')
    d.text((X0*S,(y+30)*S),msg,font=f('SemiBold',15),fill=TX)
    d.text((X0*S,(y+54)*S),'Orange: kept  ·  dark orange: scored and rejected  ·  grey: never looked at',font=f('Regular',12),fill=MU)
    d.text((48*S,(H-30)*S),FOOT,font=f('Regular',11),fill=MU)
    return im.resize((W,H),Image.LANCZOS)
fr=[];du=[]
for st in range(6):
    fr.append(tree_frame(st,0));du.append(650);fr.append(tree_frame(st,1));du.append(1000)
fr.append(tree_frame(6,1));du.append(3500)
fr[0].save('tree_replay.gif',save_all=True,append_images=fr[1:],duration=du,loop=0,optimize=True)
fr[-1].save('tree_replay_poster.png')
# ---------- B: init vs trained grid ----------
W,H=1200,720;C=2;COLS=128
def grid(d,x0,y0,st,col):
    end=st['end'];rows=(end+COLS-1)//COLS
    for r in range(128):
        pass
    d.rectangle([x0*S,y0*S,(x0+COLS*C)*S,(y0+128*C)*S],fill=(18,18,18))
    # visible keys
    full=end//COLS
    d.rectangle([x0*S,y0*S,(x0+COLS*C)*S,(y0+full*C)*S],fill=CELL)
    if end%COLS:d.rectangle([x0*S,(y0+full*C)*S,(x0+(end%COLS)*C)*S,(y0+(full+1)*C)*S],fill=CELL)
    for k in st['selected_keys']:
        r,c=divmod(k,COLS);d.rectangle([(x0+c*C)*S,(y0+r*C)*S,(x0+c*C+C)*S-1,(y0+r*C+C)*S-1],fill=col)
def inset(d,x0,y0,st,col,N=1024,cols=64,c=6):
    end=st['end'];a=end-N
    for i in range(N):
        r,q=divmod(i,cols);k=a+i
        fill=col if k in st['_sel'] else CELL
        d.rectangle([(x0+q*c)*S,(y0+r*c)*S,(x0+q*c+c)*S-2,(y0+r*c+c)*S-2],fill=fill)
def grid_frame(i):
    im=Image.new('RGB',(W*S,H*S),BG);d=ImageDraw.Draw(im)
    header(d,'Same query, different keys',f'Original initialization vs trained references on identical saved queries and keys · snapshot {i+1}/8 (not consecutive generation steps)')
    for j,(k,name,col) in enumerate([('untrained','Original initialization',(170,170,170)),('trained','Trained references',OR)]):
        st=t['states'][k][i];st['_sel']=set(st['selected_keys']);x0=70+j*560;y0=180
        far=sum(1 for q in st['selected_keys'] if q<st['end']-1024)
        d.text((x0*S,(y0-36)*S),name,font=f('Bold',19),fill=col)
        grid(d,x0,y0,st,col)
        d.text((x0*S,(y0+258)*S),f'All {st["end"]:,} visible keys · 1 cell = 1 key',font=f('Regular',12),fill=MU)
        d.text((x0*S,(y0+290)*S),'Most recent 1,024 keys, magnified',font=f('SemiBold',12),fill=TX)
        inset(d,x0,y0+310,st,col)
        d.text((x0*S,(y0+418)*S),f"{len(st['selected_keys'])} / {st['end']:,} keys",font=f('Bold',24),fill=col)
        d.text((x0*S,(y0+452)*S),f"True dense mass retained: {st['dense_mass_retained']*100:.1f}%  ·  {far} older key{"s" if far!=1 else ""} (incl. the sink)",font=f('Regular',14),fill=TX)
    d.text((48*S,(H-30)*S),FOOT,font=f('Regular',11),fill=MU)
    return im.resize((W,H),Image.LANCZOS)
fr=[grid_frame(i) for i in range(8)]
fr[0].save('query_replay.gif',save_all=True,append_images=fr[1:],duration=1800,loop=0,optimize=True)
fr[0].save('query_replay_poster.png')
