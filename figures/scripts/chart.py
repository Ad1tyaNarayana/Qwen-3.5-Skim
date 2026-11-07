import json,sys,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if 'Inter' in f: fm.fontManager.addfont(f)
plt.rcParams['font.family']='Inter'
d=json.load(open(sys.argv[1]))['scopes']['all']['configurations']
BG='#0a0a0a';OR='#ff6b1a';GR='#8a8a8a';TX='#f2efe9';MU='#7d7a75';GRID='#1f1f1f'
fig=plt.figure(figsize=(10,6.25),dpi=200,facecolor=BG)
ax=fig.add_axes([0.08,0.17,0.6,0.58],facecolor=BG)
dense=d['dense']['point']['score']*100
ax.axhline(dense,color=TX,lw=1,ls=(0,(4,4)),alpha=.6)
ax.text(1.25,dense+.45,f'Dense  {dense:.2f}%  ·  100% of keys',color=TX,alpha=.75,fontsize=9)
lab={'init_090':('Untrained  p=.90',GR,'o',(10,-4)),'trained_090':('Trained  p=.90',OR,'o',(10,-4)),
     'init_095':('Untrained  p=.95',GR,'s',(-104,-18)),'trained_095':('Trained  p=.95',OR,'s',(10,8))}
for k,(name,c,m,off) in lab.items():
    p=d[k]['point'];iv=d[k]['intervals']
    x=p['decode_key_share']*100;y=p['score']*100
    ax.errorbar(x,y,yerr=[[y-iv['score'][0]*100],[iv['score'][1]*100-y]],
        xerr=[[x-iv['decode_key_share'][0]*100],[iv['decode_key_share'][1]*100-x]],
        fmt=m,color=c,ms=9,lw=1.6,capsize=3,mec=BG,mew=1.2,zorder=3)
    ax.annotate(name,(x,y),xytext=off,textcoords='offset points',color=c,fontsize=10,fontweight='semibold')
# arrow init->trained at .90
a=d['init_090']['point'];b=d['trained_090']['point']
ax.annotate('',xy=(b['decode_key_share']*100+.12,b['score']*100-2.9),xytext=(a['decode_key_share']*100-.05,a['score']*100+1.2),
    arrowprops=dict(arrowstyle='-|>',color=OR,lw=1.3,alpha=.7,connectionstyle='arc3,rad=-.15'))
ax.set_xlim(1.2,7.3);ax.set_ylim(74,100)
ax.set_xlabel('Share of keys read during generation (%)',color=TX,fontsize=10.5,labelpad=8)
ax.set_ylabel('Answer score (%)',color=TX,fontsize=10.5,labelpad=8)
ax.tick_params(colors=MU,labelsize=9.5,length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.grid(color=GRID,lw=.8);ax.set_axisbelow(True)
fig.text(.08,.93,'QWEN3.5-4B  /  ZHA-STYLE ROUTER',color=OR,fontsize=9.5,fontweight='bold')
fig.text(.08,.865,'Training made the router read less and answer more',color=TX,fontsize=19,fontweight='bold')
fig.text(.08,.82,'96 paired synthetic prompts at ~16K context · RTX 5070 Ti',color=MU,fontsize=10.5)
diff=11.927
fig.text(.73,.66,'AT p = .90',color=MU,fontsize=9.5,fontweight='bold')
fig.text(.73,.585,f'+{diff:.1f} pts',color=OR,fontsize=30,fontweight='bold')
fig.text(.73,.545,'answer score, trained vs untrained',color=TX,fontsize=9.5)
fig.text(.73,.515,'paired 95% CI: +7.3 to +16.7',color=MU,fontsize=9)
rel=(1-b['decode_key_share']/a['decode_key_share'])*100
fig.text(.73,.39,f'−{rel:.0f}%',color=OR,fontsize=30,fontweight='bold')
fig.text(.73,.35,'keys read during generation',color=TX,fontsize=9.5)
fig.text(.73,.32,f"{a['decode_key_share']*100:.2f}% → {b['decode_key_share']*100:.2f}% of visible keys",color=MU,fontsize=9)
fig.text(.08,.03,'Bars: 95% prompt-stratified paired bootstrap intervals. Key share is not speedup or memory savings. One router checkpoint.',color=MU,fontsize=8)
fig.savefig(sys.argv[2],facecolor=BG)
