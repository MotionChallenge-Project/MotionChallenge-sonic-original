"""Compare batched SONIC observations/history/actions with the source adapter."""
import argparse,json,sys,types
from pathlib import Path
import numpy as np
import torch
from onnx.reference import ReferenceEvaluator
from motion_challenge.sdk import DeployableStateBatch,DeployableStatePacket,MotionReferenceBatch,ReferenceFrame
from motion_challenge.simulation.tensor_reference import TensorMotionBank
from motion_challenge.simulation.tensor_math import rotate
from tensor_adapter import TensorSONIC


def validate(prepared,weights,source,encoder,device):
    root=Path(__file__).resolve().parent
    runtime=json.loads((root/"runtime.mjwarp.json").read_text());cfg=json.loads((root/"policy.json").read_text())
    meta=json.loads((Path(prepared)/"dataset.json").read_text());names=tuple(meta["joint_names"])
    slots=7;bank=TensorMotionBank(np.array(np.load(Path(prepared)/"poses.npy",mmap_mode="r")),meta["starts"],meta["lengths"],np.full(len(meta["names"]),meta["fps"]),meta["names"],device)
    tensor=TensorSONIC(root,runtime,names,slots,device,weights);tensor.reset(torch.arange(slots,device=device))
    sys.modules.setdefault("onnxruntime",types.ModuleType("onnxruntime"))
    from adapter import SONIC
    from motion_challenge.adapters.onnx import vector
    decoder_ref=ReferenceEvaluator(str(source));encoder_ref=ReferenceEvaluator(str(encoder or source))
    originals=[]
    for _ in range(slots):
        a=SONIC.__new__(SONIC);a.cfg=cfg;a.runtime=runtime;a.names=cfg["policy_joint_names"];a.default=vector(cfg["default_joint_pos"],a.names)
        if "encoder_dimension" in cfg:a.encoder=encoder_ref;a.encoder_dim=cfg["encoder_dimension"]
        originals.append(a)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(19119);maximum=0.;action_error=0.;token_error=0.;target_error=0.
    for step in range(24):
        request=torch.tensor(np.arange(slots)*.031+step*.02,device=device,dtype=torch.float64)[:,None]+torch.tensor(runtime["reference_offsets_s"],device=device,dtype=torch.float64)[None]
        ids=torch.arange(slots,device=device)[:,None].expand_as(request)
        t,p,q,j,linear,omega,jv=bank.state(ids,request)
        refs=MotionReferenceBatch(request,t,j,p,q,jv,linear,rotate(q,omega))
        joints=rng.normal(0,.1,(slots,29));vel=rng.normal(0,.1,(slots,29));quat=rng.normal(size=(slots,4));quat/=np.linalg.norm(quat,axis=-1,keepdims=True);angular=rng.normal(0,.1,(slots,3))
        if step==11:tensor.reset(torch.tensor([1,4],device=device))
        state=DeployableStateBatch(names,torch.full((slots,),step*.02,device=device),*[torch.tensor(x,device=device) for x in (joints,vel,quat,angular)])
        g1,proprio=tensor.observations(state,refs)
        cpu={n:getattr(refs,n).cpu().numpy() for n in ("requested_time_s","source_time_s","joint_position_rad","root_position_world_m","root_orientation_xyzw","joint_velocity_rad_s","root_linear_velocity_world_m_s","root_angular_velocity_world_rad_s")}
        expected_actions=[];expected_tokens=[]
        for row,a in enumerate(originals):
            packet=DeployableStatePacket(step*.02,names,tuple(joints[row]),tuple(vel[row]),tuple(quat[row]),tuple(angular[row]))
            frames=[ReferenceFrame(*[tuple(cpu[n][row,k]) if cpu[n][row,k].ndim else float(cpu[n][row,k]) for n in cpu]) for k in range(10)]
            if step==0 or (step==11 and row in [1,4]):a.reset(packet,frames,0)
            expected=a.observations(packet,frames)
            if "encoder_dimension" in cfg:
                raw=a.encoder_observations(packet,frames)
                rotation_start=601 if cfg["encoder_dimension"]==1762 else 584
                expected_g1=np.concatenate((raw[0,4:584],raw[0,rotation_start:rotation_start+60]))
                expected_proprio=expected["obs_dict"][0,64:]
                token=expected["obs_dict"][0,:64]
                action=decoder_ref.run(["action"],expected)[0].reshape(-1)
            else:
                expected_g1=expected["g1_input"].astype(np.float32);expected_proprio=expected["proprioception"].astype(np.float32)
                action,token=decoder_ref.run(["action","token"],{k:v.astype(np.float32) for k,v in expected.items()});action=action.reshape(-1);token=token.reshape(-1)
            for actual,want in [(g1[row].cpu().numpy(),expected_g1),(proprio[row].cpu().numpy(),expected_proprio)]:
                maximum=max(maximum,float(abs(actual-want).max()));np.testing.assert_allclose(actual,want,atol=2e-6,rtol=2e-6)
            expected_actions.append(action);expected_tokens.append(token)
            a.last=action.astype(np.float32)
        with torch.inference_mode():actions,tokens=tensor.network(g1,proprio)
        expected_actions=np.stack(expected_actions);expected_tokens=np.stack(expected_tokens)
        action_error=max(action_error,float(abs(actions.cpu().numpy()-expected_actions).max()))
        token_error=max(token_error,float(abs(tokens.cpu().numpy()-expected_tokens).max()))
        np.testing.assert_array_equal(tokens.cpu().numpy(),expected_tokens)
        np.testing.assert_allclose(actions.cpu().numpy(),expected_actions,atol=5e-5,rtol=2e-5)
        targets=tensor.default.cpu().numpy()+tensor.scale.cpu().numpy()*actions.cpu().numpy()
        expected_targets=tensor.default.cpu().numpy()+tensor.scale.cpu().numpy()*expected_actions
        target_error=max(target_error,float(abs(targets-expected_targets).max()))
        tensor.last.copy_(torch.tensor(expected_actions,device=device))
    report={"status":"passed","steps":24,"batch":slots,"partial_reset_step":11,"partial_reset_slots":[1,4],"observation_max_abs_error":maximum,"network_max_abs_error":action_error,"token_max_abs_error":token_error,"joint_target_max_abs_error_rad":target_error,"history_action_inputs":"identical source-network outputs"}
    (root/"validation/tensor-observation-parity.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--prepared",required=True,type=Path);p.add_argument("--weights",required=True,type=Path);p.add_argument("--source",required=True,type=Path);p.add_argument("--encoder",type=Path);p.add_argument("--device",default="cuda:0")
    args=p.parse_args();validate(**vars(args))
