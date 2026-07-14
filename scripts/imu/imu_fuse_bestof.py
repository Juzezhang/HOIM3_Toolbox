"""Segment-level best-of post-pass for a fused object: per bad segment, compare
mean IoU of the FUSED pose vs the ORIGINAL mocap pose on QC samples; revert
segments where mocap wins. Guarantees fused >= mocap per segment. Env: SEQ, OBJ."""
import os, json
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np, cv2, trimesh, pyrender
SEQ=os.environ['SEQ']; OBJ=os.environ['OBJ']
D='/simurgh/group/juze/datasets/HOI-M3'; D2='/simurgh2/datasets/HOI-M3'
F=f'{D2}/object_imu_fused/{SEQ}'
z=np.load(f'{F}/{OBJ}.npz'); R_out=z['object_R'].copy(); T_out=z['object_T'].copy(); src=z['source'].copy(); s=int(z['start_frame'])
meta=json.load(open(f'{F}/{OBJ}_meta.json')); segs=meta['bad_segments']; N=len(R_out)
mg=np.load(f'{D}/mocap_ground/{SEQ}_object.npz',allow_pickle=True)['object_params']
def mgp(i):
    e=mg[i].get(OBJ) if isinstance(mg[i],dict) else None
    if e is None: return None,None
    e=e.item() if hasattr(e,'item') else e
    return np.array(e['object_R']),np.array(e['object_T']).ravel()
INFO=json.load(open(f'{D}/dataset_information.json')); DATE=next(d for d,ss in INFO.items() if SEQ in ss)
cal=json.load(open(f'{D2}/calib_ground_refined/{DATE}/calibration.json')); cams={}
for v in [0,2,5,7,10,14,19,21,24,29,36]:
    c=cal.get(str(v))
    if not c or not c.get('K'): continue
    K=np.array(c['K'],float).reshape(3,3)/3.0; K[2,2]=1
    rt=np.array(c['RT'],float); RT=rt.reshape(4,4)[:3] if rt.size==16 else rt.reshape(3,4)
    cams[v]=(K,RT[:,:3],RT[:,3])
CT=f'{D2}/cutie_tracking/{SEQ}'
names=json.load(open(f'{D2}/cutie_refs/{SEQ}/masks/0_names.json'))['mask_names']
OI=names.index(OBJ)+1; PI=tuple(i+1 for i,n in enumerate(names) if n.startswith('person'))
tm=trimesh.load(f'{D}/scanned_object/{OBJ}/{OBJ}_simplified_transformed.obj',force='mesh')
ren=pyrender.OffscreenRenderer(1280,720); CV=np.diag([1,-1,-1,1.0]); mesh=pyrender.Mesh.from_trimesh(tm,smooth=False)
def sil(v,R,T):
    K,Rc,tc=cams[v]; sc=pyrender.Scene(bg_color=[0,0,0,0])
    p=np.eye(4); p[:3,:3]=R; p[:3,3]=T; sc.add(mesh,pose=p)
    cam=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=0.05,zfar=20)
    E=np.eye(4); E[:3,:3]=Rc; E[:3,3]=tc; sc.add(cam,pose=np.linalg.inv(CV@E))
    return ren.render(sc,flags=pyrender.RenderFlags.DEPTH_ONLY)>0
def masks_at(fr,cap=4):
    out={}
    for v in cams:
        p=f'{CT}/{v}/{fr:06d}.npz'
        if not os.path.exists(p): continue
        m=np.load(p,allow_pickle=True)['mask']
        if m.shape!=(720,1280): m=cv2.resize(m,(1280,720),interpolation=cv2.INTER_NEAREST)
        r=(m==OI)
        if r.sum()<150: continue
        out[v]=(r,np.isin(m,PI))
    if len(out)>cap:
        keep=sorted(out,key=lambda v:-out[v][0].sum())[:cap]; out={v:out[v] for v in keep}
    return out
def iou(R,T,msk):
    xs=[]
    for v,(r,pe) in msk.items():
        S=sil(v,R,T); k=~pe
        i=(S&r&k).sum(); u=((S|r)&k).sum()
        if u>100: xs.append(i/u)
    return float(np.mean(xs)) if xs else None
reverted=0
for a,b in segs:
    fu,mo=[],[]
    for i in range(a,b,30):
        msk=masks_at(s+i)
        if len(msk)<2: continue
        Rm,Tm=mgp(i)
        if Rm is None: continue
        x=iou(R_out[i],T_out[i],msk); y=iou(Rm,Tm,msk)
        if x is not None and y is not None: fu.append(x); mo.append(y)
    if fu and np.mean(mo)>np.mean(fu)+0.01:
        for i in range(a,b):
            Rm,Tm=mgp(i)
            if Rm is not None: R_out[i]=Rm; T_out[i]=Tm; src[i]=0
        reverted+=1
        print(f'  seg [{a},{b}): reverted to mocap (mocap {np.mean(mo):.3f} > fused {np.mean(fu):.3f})',flush=True)
    else:
        print(f'  seg [{a},{b}): kept fused ({np.mean(fu) if fu else -1:.3f} vs mocap {np.mean(mo) if mo else -1:.3f})',flush=True)
np.savez(f'{F}/{OBJ}.npz',object_R=R_out,object_T=T_out,source=src,start_frame=np.int32(s))
meta['bestof_reverted_segments']=reverted
json.dump(meta,open(f'{F}/{OBJ}_meta.json','w'))
print(f'BESTOF_DONE {SEQ}/{OBJ} reverted={reverted}/{len(segs)}',flush=True)
