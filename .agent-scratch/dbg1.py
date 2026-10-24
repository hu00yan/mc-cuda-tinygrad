import os,sys,time
os.environ['DEV']='CPU'; os.environ['PARALLEL']='0'
sys.path.insert(0,'.')
import numpy as np
t=time.time()
from tinygrad import Tensor, dtypes
from minecu_tg import generate_world
from minecu_tg.render import camera_basis
from minecu_tg_opt.render import render_variant
from minecu_tg_opt.lm38 import render_arm_c
print("imports", round(time.time()-t,1), flush=True)
world,ground,batch,res,steps=24,6,2,32,32
t=time.time(); one=generate_world(world,ground).numpy(); print("world",round(time.time()-t,1),flush=True)
vox=Tensor(np.ascontiguousarray(np.stack([one]*batch)),dtype=dtypes.int8)
cams=np.array([[world/2.0,float(ground),world/2.0,0.05*(1 if i%2==0 else -1),0.0] for i in range(batch)],dtype=np.float32)
t=time.time(); basis=camera_basis(Tensor(cams,dtype=dtypes.float32),70.0,res,res); print("basis",round(time.time()-t,1),flush=True)
t=time.time(); o,s=render_variant(vox,basis,res,res,steps,"flat_gather",count_steps=True); o=o.numpy(); s=s.numpy(); print("render",round(time.time()-t,1),o.shape,s.mean(),flush=True)
t=time.time(); o2,s2=render_arm_c(vox,basis,res,res,steps,"fg_late_addr",count_steps=True); o2=o2.numpy(); s2=s2.numpy(); print("armc",round(time.time()-t,1),np.array_equal(o,o2),np.array_equal(s,s2),flush=True)
