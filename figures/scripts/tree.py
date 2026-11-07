from PIL import Image,ImageDraw,ImageFont
import random
W,H=1200,720;S=2
BG=(10,10,10);OR=(255,107,26);TX=(242,239,233);MU=(125,122,117);DIM=(48,48,48);WH=(255,255,255)
F='/usr/share/fonts/opentype/inter/Inter-%s.otf'
def f(w,s):return ImageFont.truetype(F%w,s*S)
random.seed(7)
# tree: node=(level,index); span at level L width 4**L keys (level1=4 keys)
N=16360
levels=list(range(6,0,-1))
rows=[];kept={}
cand=[(6,i) for i in range(N//4096)]  # 3 complete top nodes (+ orphan handled loosely)
for L in levels:
    cand=[c for c in cand if (c[1]+1)*4**L<=N]
    k=max(1,min(len(cand)//3+1,3)) if L>1 else min(4,len(cand))
    ks=sorted(random.sample(range(len(cand)),k))
    keep=[cand[i] for i in ks]
    rows.append((L,cand,keep))
    cand=[(L-1,c[1]*4+j) for c in keep for j in range(4)]
X0,X1=230,1000
ytop=150;dy=72
pos={}
for r,(L,cand,keep) in enumerate(rows):
    n=len(cand);y=ytop+r*dy
    for i,c in enumerate(cand):
        pos[c]=(X0+(X1-X0)*(i+.5)/n,y)
barY=ytop+6*dy+30
def frame(stage,sub):
    im=Image.new('RGB',(W*S,H*S),BG);d=ImageDraw.Draw(im)
    d.text((60*S,40*S),'QWEN3.5-4B  /  ZHA-STYLE ROUTER',font=f('Bold',13),fill=OR)
    d.text((60*S,62*S),'Searching the context with a tree',font=f('Bold',30),fill=TX)
    d.text((60*S,104*S),'Score a few summaries, keep the likely ones, descend only into those',font=f('Regular',15),fill=MU)
    for r,(L,cand,keep) in enumerate(rows):
        y=ytop+r*dy
        d.text((1035*S,(y-12)*S),f'Level {L}',font=f('SemiBold',15),fill=TX if r<=stage else MU)
        d.text((1035*S,(y+6)*S),f'{4**L:,}-key spans',font=f('Regular',12),fill=MU)
        if r>stage:continue
        for c in cand:
            x,yy=pos[c]
            if r>0:
                par=(c[0]+1,c[1]//4);px,py=pos[par]
                d.line([(px*S,(py+7)*S),(x*S,(yy-7)*S)],fill=(80,45,20) if (r<stage or sub) and c in keep else (40,40,40),width=S)
        for c in cand:
            x,yy=pos[c];rad=7
            if r<stage or sub:
                col=OR if c in keep else DIM
            else: col=WH
            d.ellipse([(x-rad)*S,(yy-rad)*S,(x+rad)*S,(yy+rad)*S],fill=col)
    # bar
    d.rectangle([X0*S,barY*S,X1*S,(barY+22)*S],fill=(28,28,28))
    d.text((1035*S,(barY+2)*S),'Original keys',font=f('SemiBold',15),fill=TX)
    sc=lambda k:X0+(X1-X0)*k/N
    def span(a,b,col):d.rectangle([sc(a)*S,barY*S,max(sc(b),sc(a)+2)*S,(barY+22)*S],fill=col)
    span(0,4,OR);span(N-4,N,OR)
    if stage>=6:
        for c in rows[-1][2]:
            a=1+c[1]*4;span(a,a+4,OR)
    # left legend
    lab=['SCORE','KEEP','SKIP'];cols=[WH,OR,MU];desc=['candidate summaries','top-p branches','rejected branches']
    for i in range(3):
        d.text((60*S,(300+i*70)*S),lab[i],font=f('Bold',13),fill=cols[i])
        d.text((60*S,(318+i*70)*S),desc[i],font=f('Regular',14),fill=MU)
    step='Gather selected keys + sink + recent chunk, then attend' if stage>=6 else (f'Level {rows[stage][0]}: '+('keep the smallest set covering p = .90 of estimated mass' if sub else 'score candidate summaries against the query'))
    d.text((X0*S,(barY+42)*S),step,font=f('SemiBold',16),fill=TX)
    d.text((60*S,(H-34)*S),'Schematic illustration of the routing algorithm (C4 / L6, p = .90), not a replay of recorded decisions. Node spacing not to scale.',font=f('Regular',12),fill=MU)
    return im.resize((W,H),Image.LANCZOS)
frames=[];dur=[]
for s in range(6):
    frames.append(frame(s,0));dur.append(700)
    frames.append(frame(s,1));dur.append(900)
frames.append(frame(6,1));dur.append(3000)
frames[0].save('tree_search.gif',save_all=True,append_images=frames[1:],duration=dur,loop=0)
frames[-1].save('tree_search_final.png')
print([ (L,len(c),len(k)) for L,c,k in rows])
