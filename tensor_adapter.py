"""SONIC G1 batched reference features, ten-frame history and PD commands."""
import json,re
from pathlib import Path
import torch
from sonic_torch import SonicNetwork
from motion_challenge.simulation.tensor_math import normalize,conjugate,multiply,rotate,matrix


def vector(value,names,device):
    if isinstance(value,dict):
        result=[]
        for name in names:
            matches=[v for pattern,v in value.items() if re.fullmatch(pattern,name)]
            if len(matches)>1:raise ValueError("Ambiguous joint pattern")
            result.append(matches[0] if matches else 0.)
        value=result
    t=torch.as_tensor(value,device=device,dtype=torch.float64)
    return t.expand(len(names)) if t.ndim==0 else t


class TensorSONIC:
    def __init__(self,root,runtime,joint_names,slots,device,converted_weights):
        self.device=torch.device(device)
        self.cfg=cfg=json.loads((Path(root)/"policy.json").read_text())
        self.released="encoder_dimension" in cfg
        names=cfg["policy_joint_names"]
        self.order=[joint_names.index(n) for n in names];self.inverse=[names.index(n) for n in joint_names]
        self.default=vector(cfg["default_joint_pos"],names,device)
        self.scale=vector(cfg["action_scale"],names,device)
        self.kp=vector(cfg["joint_kp"],names,device)[self.inverse]
        self.kd=vector(cfg["joint_kd"],names,device)[self.inverse]
        self.clip=cfg.get("action_clip")
        self.network=SonicNetwork(converted_weights,device).eval()
        if self.network.metadata["decoder_sha256"]!=runtime["weights_sha256"]:raise ValueError("Source weight identity differs")
        self.history=[torch.zeros((slots,10,n),device=device,dtype=torch.float64) for n in (3,29,29,29,3)]
        self.last=torch.zeros((slots,29),device=device)
        self.align=torch.zeros((slots,4),device=device,dtype=torch.float64)
        self.fresh=torch.ones(slots,device=device,dtype=torch.bool)

    def heading(self,q):
        if self.released:
            m=matrix(q);a=torch.atan2(m[...,1,0],m[...,0,0])/2
            z=torch.zeros_like(a);return torch.stack((z,z,a.sin(),a.cos()),dim=-1)
        zw=q[...,2:];norm=torch.linalg.vector_norm(zw,dim=-1,keepdim=True)
        safe=zw/norm.clamp_min(1e-12)
        identity=torch.zeros_like(safe);identity[...,1]=1
        safe=torch.where(norm<1e-12,identity,safe)
        return torch.cat((torch.zeros_like(safe),safe),dim=-1)

    def reset(self,slots):
        self.last[slots]=0;self.fresh[slots]=True

    def observations(self,state,reference):
        q=state.joint_position_rad[:,self.order].double();qd=state.joint_velocity_rad_s[:,self.order].double()
        actual=normalize(state.root_orientation_xyzw.double());inv=conjugate(actual)
        omega=rotate(inv,state.root_angular_velocity_world_rad_s.double())
        gravity=rotate(inv,torch.tensor([0.,0.,-1.],device=self.device,dtype=torch.float64))
        rotations=normalize(reference.root_orientation_xyzw.double())
        align=multiply(self.heading(actual),conjugate(self.heading(rotations[:,0])))
        self.align.copy_(torch.where(self.fresh[:,None],align,self.align))
        terms=(omega,q-self.default,qd,self.last,gravity)
        for i,(history,value) in enumerate(zip(self.history,terms)):
            if not self.released:value=value.float().double()
            if self.released:
                initial=torch.zeros_like(history)
                if i==4:initial[:,:,2]=1
            else:initial=value[:,None].expand_as(history)
            history.copy_(torch.where(self.fresh[:,None,None],initial,history))
            history[:,:-1]=history[:,1:].clone();history[:,-1]=value
        self.fresh.zero_()
        left=self.heading(actual) if self.cfg.get("reference_orientation_frame")=="heading" else actual
        rot6=matrix(multiply(multiply(conjugate(left),self.align)[:,None],rotations))[...,:,:2]
        rq=reference.joint_position_rad[:,:,self.order].double()
        rv=reference.joint_velocity_rad_s[:,:,self.order].double()
        return (torch.cat((rq.flatten(1),rv.flatten(1),rot6.flatten(1)),dim=-1).float(),
                torch.cat([h.flatten(1) for h in self.history],dim=-1).float())

    @torch.inference_mode()
    def act(self,state,reference):
        raw,_=self.network(*self.observations(state,reference))
        action=raw.clamp(-self.clip,self.clip) if self.clip is not None else raw
        self.last.copy_(raw if self.released else action)
        target=(self.default+self.scale*action)[:,self.inverse]
        zero=torch.zeros_like(target)
        return torch.stack((target,zero,self.kp.expand_as(target),self.kd.expand_as(target),zero),dim=1).float()
