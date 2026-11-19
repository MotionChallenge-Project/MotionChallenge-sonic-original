"""Convert and independently verify this repository's SONIC checkpoints."""
import json
from pathlib import Path
import numpy as np
import torch
from onnx.reference import ReferenceEvaluator
from sonic_torch import SonicNetwork,convert
from motion_challenge.validation import sha256_file


def prepare(root,runtime,source,output,artifacts,device):
    root,source,output=Path(root),Path(source),Path(output)
    cfg=json.loads((root/runtime["inference_config"]).read_text())
    encoder=artifacts.get("encoder",source)
    encoder_hash=runtime.get("artifacts",{}).get("encoder",{}).get("sha256",runtime["weights_sha256"])
    report_path=output/"parity.json"
    if report_path.exists() and (output/"graph.json").exists():
        previous=json.loads(report_path.read_text())
        if previous.get("converter_sha256")==sha256_file(root/"sonic_torch.py") and previous.get("hook_sha256")==sha256_file(__file__) and previous.get("encoder_source_sha256")==encoder_hash and previous.get("decoder_source_sha256")==runtime["weights_sha256"] and previous.get("graph_sha256")==sha256_file(output/"graph.json"):
            SonicNetwork(output,"cpu")
            return output
    convert(encoder,source,output,encoder_hash,runtime["weights_sha256"])
    er=ReferenceEvaluator(str(encoder));dr=ReferenceEvaluator(str(source))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(202619);cases=[];maximum={"token":0.,"action":0.}
    for batch in (1,7,40,256):
        inputs=rng.normal(0,.3,(batch,640)).astype(np.float32)
        proprio=rng.normal(0,.3,(batch,930)).astype(np.float32)
        expected_token=[];expected_action=[]
        for row in range(batch):
            if "encoder_dimension" in cfg:
                obs=np.zeros((1,cfg["encoder_dimension"]),np.float32)
                rotation_start=601 if cfg["encoder_dimension"]==1762 else 584
                obs[:,4:584]=inputs[row:row+1,:580];obs[:,rotation_start:rotation_start+60]=inputs[row:row+1,580:]
                token=er.run(["encoded_tokens"],{"obs_dict":obs})[0]
                action=dr.run(["action"],{"obs_dict":np.concatenate((token,proprio[row:row+1]),axis=1)})[0]
            else:
                action,token=er.run(["action","token"],{"g1_input":inputs[row],"proprioception":proprio[row]})
            expected_token.append(token.reshape(-1));expected_action.append(action.reshape(-1))
        expected_token=np.stack(expected_token);expected_action=np.stack(expected_action)
        for target_device in dict.fromkeys(("cpu",str(device))):
            network=SonicNetwork(output,target_device).eval()
            with torch.inference_mode():action,token=network(torch.tensor(inputs,device=target_device),torch.tensor(proprio,device=target_device))
            a,t=action.cpu().numpy(),token.cpu().numpy()
            np.testing.assert_array_equal(t,expected_token)
            np.testing.assert_allclose(a,expected_action,atol=5e-5,rtol=2e-5)
            errors={"token":float(abs(t-expected_token).max()),"action":float(abs(a-expected_action).max())}
            cases.append({"batch":batch,"device":target_device,"max_abs_error":errors})
            for key,value in errors.items():maximum[key]=max(maximum[key],value)
    report={"status":"passed","encoder_source_sha256":encoder_hash,"decoder_source_sha256":runtime["weights_sha256"],
            "reference":"original ONNX ReferenceEvaluator, native G1 mode 0","tolerance":{"token":"exact","action_atol":5e-5,"action_rtol":2e-5},
            "cases":cases,"max_abs_error":maximum,"converter_sha256":sha256_file(root/"sonic_torch.py"),
            "hook_sha256":sha256_file(__file__),"graph_sha256":sha256_file(output/"graph.json")}
    report_path.write_text(json.dumps(report,indent=2)+"\n")
    return output
