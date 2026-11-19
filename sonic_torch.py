"""Batched SONIC G1 encoder, scalar quantizer and decoder."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from motion_challenge.validation import sha256_file


class SonicNetwork(torch.nn.Module):
    def __init__(self, directory, device="cpu"):
        super().__init__()
        directory=Path(directory)
        self.metadata=json.loads((directory/"graph.json").read_text())
        if sha256_file(directory/"weights.safetensors")!=self.metadata["weights_sha256"]:
            raise ValueError("Converted weights checksum mismatch")
        tensors=load_file(str(directory/"weights.safetensors"),device=device)
        for name,value in tensors.items():self.register_buffer(name,value)

    def encode(self,g1_input):
        batch=len(g1_input)
        indices=self.encoder_input_indices
        x=g1_input[:,indices.clamp_min(0)]*self.encoder_input_mask
        for i in range(self.metadata["encoder_layers"]):
            x=F.linear(x,getattr(self,f"encoder_{i}_weight"),getattr(self,f"encoder_{i}_bias"))
            if i+1<self.metadata["encoder_layers"]:x=x*torch.sigmoid(x)
        x=x.reshape(batch,2,1,32)
        x=torch.tanh(x+self.quant_offset)*self.quant_scale-self.quant_shift
        # Preserve the author's straight-through quantizer's inference arithmetic.
        x=(x+(torch.round(x)-x))/self.quant_divisor
        return x.reshape(batch,64)

    def decode(self,tokens,proprioception):
        x=torch.cat((tokens,proprioception),dim=-1)
        for i in range(self.metadata["decoder_layers"]):
            x=x@getattr(self,f"decoder_{i}_weight")+getattr(self,f"decoder_{i}_bias")
            if i+1<self.metadata["decoder_layers"]:x=x*torch.sigmoid(x)
        return x

    def forward(self,g1_input,proprioception):
        tokens=self.encode(g1_input)
        return self.decode(tokens,proprioception),tokens


def convert(encoder,decoder,directory,encoder_sha256,decoder_sha256):
    import onnx
    from onnx import numpy_helper,helper
    for path,digest in [(encoder,encoder_sha256),(decoder,decoder_sha256)]:
        if sha256_file(path)!=digest:raise ValueError("Source checkpoint hash differs")
    em=onnx.load(encoder);dm=onnx.load(decoder)
    ei={x.name:numpy_helper.to_array(x).copy() for x in em.graph.initializer}
    di={x.name:numpy_helper.to_array(x).copy() for x in dm.graph.initializer}
    weights={};ecount=0;dcount=0
    import numpy as np
    from onnx.reference import ReferenceEvaluator
    from onnx.utils import Extractor
    source_input=em.graph.input[0]
    input_shape=[d.dim_value for d in source_input.type.tensor_type.shape.dim]
    em.graph.value_info.append(helper.make_tensor_value_info("/g1/Reshape_output_0",onnx.TensorProto.FLOAT,[1,640]))
    pruned=Extractor(em).extract_model([source_input.name],["/g1/Reshape_output_0"])
    probe=np.arange(np.prod(input_shape),dtype=np.float32).reshape(input_shape)
    mapping=ReferenceEvaluator(pruned).run(None,{source_input.name:probe})[0].reshape(-1).astype(np.int64)
    if len(input_shape)==1:
        compact=mapping
    else:
        compact=np.full(640,-1,np.int64)
        rotation_start=601 if input_shape[-1]==1762 else 584
        for i,index in enumerate(mapping):
            if 4<=index<584:compact[i]=index-4
            elif rotation_start<=index<rotation_start+60:compact[i]=index-rotation_start+580
    weights["encoder_input_indices"]=torch.from_numpy(compact)
    weights["encoder_input_mask"]=torch.from_numpy((compact>=0).astype(np.float32))
    for node in em.graph.node:
        if node.op_type=="Gemm" and node.name.startswith("/g1/module/"):
            attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
            if attrs!={"alpha":1.,"beta":1.,"transB":1}:raise ValueError("Unexpected encoder linear convention")
            weights[f"encoder_{ecount}_weight"]=torch.tensor(ei[node.input[1]])
            weights[f"encoder_{ecount}_bias"]=torch.tensor(ei[node.input[2]])
            ecount+=1
        if node.op_type=="Constant":
            constants={"/quantizer/Constant_1_output_0":"quant_offset","/quantizer/Constant_2_output_0":"quant_scale","/quantizer/Constant_3_output_0":"quant_shift","/quantizer/Constant_4_output_0":"quant_divisor"}
            if node.output[0] in constants:
                weights[constants[node.output[0]]]=torch.tensor(numpy_helper.to_array(helper.get_attribute_value(node.attribute[0])).copy())
    for node in dm.graph.node:
        if node.op_type=="MatMul" and node.name.startswith("/g1_dyn/module/"):
            weights[f"decoder_{dcount}_weight"]=torch.tensor(di[node.input[1]])
            layer=int(node.name.split("module.")[1].split("/")[0])
            weights[f"decoder_{dcount}_bias"]=torch.tensor(di[f"module.decoders.g1_dyn.module.{layer}.bias"])
            dcount+=1
    if ecount!=5 or dcount not in (7,9) or len(weights)!=2*ecount+2*dcount+6:
        raise ValueError("Unexpected SONIC G1 architecture")
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    save_file(weights,str(directory/"weights.safetensors"))
    metadata={"encoder_sha256":encoder_sha256,"decoder_sha256":decoder_sha256,
              "encoder_layers":ecount,"decoder_layers":dcount,"encoder_mode":0,
              "weights_sha256":sha256_file(directory/"weights.safetensors"),
              "converter_sha256":sha256_file(__file__)}
    (directory/"graph.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata
