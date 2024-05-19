from typing import Final, Dict, Optional, Iterable, Tuple, List

import torch
import torch.nn as nn
from torch.distributions import Distribution

from core import VType, DType, EnvObjClass, ObjectOrientedEnv, EnvInfo
from utils.typings import ObjectTensors, NamedTensors, ObjectDistributions
import utils
import alg.functional as F
from ..mask_generator import MaskGenerator


class DistributionDecoder(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, vtype: VType,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()
        self._vtype = vtype
        self._ptype = vtype.ptype
        self.transform = nn.Sequential(
            nn.Linear(dim_in, dim_hidden, device=device, dtype=dtype),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden, device=device, dtype=dtype),
            nn.LeakyReLU(),
        )
        self.sub_decoders = {
            key: nn.Linear(dim_hidden, dim_param, device=device, dtype=dtype)
            for key, dim_param in self._ptype.param_sizes.items()}
        for param, decoder in self.sub_decoders.items():
            self.add_module(f"{param} decoder", decoder)

    def forward(self, x: torch.Tensor):
        x = self.transform(x)
        params = {k: decoder(x) for k, decoder in self.sub_decoders.items()}
        out = self._ptype(**params)
        return out


class AttributeEncoder(nn.Module):

    def __init__(self, vtype: VType,
                 dim_hidden: int, dim_out: int, norm_momentum: float,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.vtype: Final = vtype
        self.dim_out: Final = dim_out

        self.input_norm = nn.BatchNorm1d(vtype.size, device=device, dtype=dtype,
                                         momentum=norm_momentum)
        self.f = nn.Sequential(
            nn.Linear(vtype.size, dim_hidden, device=device, dtype=dtype),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_out, device=device, dtype=dtype),
            nn.LeakyReLU()
        )

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        data = self.vtype.raw2input(raw)  # (n_sample, n_obj, feature)
        data: torch.Tensor = self.input_norm(data.transpose(1, 2))
        data = data.transpose(1, 2)
        return self.f(data)


class ClassAttributeEncoder(nn.Module):
    def __init__(self, c: EnvObjClass, dim_hidden: int,
                 dim_encoding: int, norm_momentum: float,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.c: Final = c
        self.fieldnames: Final = c.fieldnames()

        self.encoders = {
            fieldname: AttributeEncoder(c.field_vtypes[fieldname],
                                        dim_hidden, dim_encoding, norm_momentum,
                                        device, dtype)
            for fieldname in c.fieldnames()
        }
        for fieldname, module in self.encoders.items():
            self.add_module('%s_encoder' % fieldname, module)
        

    def forward(self, raws: NamedTensors) -> torch.Tensor:
        """
        Args:
            raw_tensors (NamedTensors): A dictionary {attribute_name: attribute_data},
                where attribute_data is a tensor as (batch_size, n_object, *shape)
        Returns:
            Tensor: object variable encodings shaped as (batch_size, n_obj, n_field, dim_out).
        """

        if len(raws) == 0:
            raise ValueError("The input dict inclues no attribute. "
                             f"Please check the definition of {self.c}")
        temp = [self.encoders[fieldname].forward(raws[fieldname])
                for fieldname in self.fieldnames]
        enc = torch.stack(temp, dim=2)
        return enc


class VariableEncoder(nn.Module):

    def __init__(self, envinfo: EnvInfo,
                 dim_hidden: int, dim_out: int, norm_momentum: float,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.encoders = {
            c.name: ClassAttributeEncoder(
                c, dim_hidden, dim_out, norm_momentum,
                device, dtype)
            for c in envinfo.classes
        }
        for cls_name, module in self.encoders.items():
            self.add_module('%s_encoder' % cls_name, module)

    def forward(self, raw_attributes: ObjectTensors) -> NamedTensors:
        return {cls_name: self.encoders[cls_name].forward(x)
                for cls_name, x in raw_attributes.items()}



class ClassEncoder(nn.Module):
    def __init__(self, c: EnvObjClass, dim_in: int, dim_local: int, dim_global: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.transform_local = nn.Linear(dim_in, dim_local, device=device, dtype=dtype)
        self.transform_global = nn.Linear(dim_in, dim_global, device=device, dtype=dtype)

        self.dim_out_local = dim_local * c.n_field()
        self.dim_out_global = dim_global * c.n_field()

    def encode_local(self, x: torch.Tensor, attrmask: torch.Tensor):
        '''
        Args:
            x: (batch_size, n_obj_c, n_field_c, dim_oin)
            attrmask: (batch_size, n_statefield_c, n_field_c)
        Returns:
            Tensor: (batchsize, n_obj_c, n_statefield_c, n_field_c * dim_local)
        '''
        
        # (batch_size, n_obj_c, n_field_c, dim_local)
        x = torch.relu(self.transform_local.forward(x))

        # (batch_size, n_obj_c, n_statefield_c, n_field_c, dim_local)
        x = F.masked_retain(x.unsqueeze(dim=2), attrmask.unsqueeze(dim=-1).unsqueeze(1))
        
        # (batch_size, n_obj_c, n_statefield_c, n_field_c*dim_local)
        x = x.flatten(start_dim=-2)

        return x

    def encode_global(self, x: torch.Tensor, attrmask: torch.Tensor,
                      objmask: Optional[torch.Tensor] = None):
        '''
        Args:
            x: (batch_size, n_obj_c, n_field_c, dim_in)
            attrmask: (batch_size, n_statefield, n_field_c)
            objmask: (batchsize, n_obj_c)
        Returns:
            Tensor: (batchsize, n_statefield + 1, n_field_c * dim_global)
        '''
        
        # (batch_size, n_obj_c, n_field_c, dim_global)
        x = self.transform_global.forward(x)

        if objmask is not None:
            x = F.masked_retain(x, objmask.reshape(*objmask.shape, 1, 1))

        # (batch_size, n_field_c, dim_global)
        x = torch.relu(torch.sum(x, dim=1))
        r = x  # encoding for reward

        # (batch_size, n_statefield, n_field_c, dim_global)
        x = F.masked_retain(x.unsqueeze(dim=1), attrmask.unsqueeze(dim=-1))

        # (batch_size, n_statefield + 1, n_field_c, dim_global)
        x = torch.cat((x, r.unsqueeze(dim=1)), dim=1)

        # (batch_size, n_statefield + 1, n_field_c*dim_global)
        x = x.flatten(start_dim=-2)

        return x

    def forward(self, x: torch.Tensor, localmask: torch.Tensor,
                globalmask: torch.Tensor,
                objmask: Optional[torch.Tensor] = None):
        out_local = self.encode_local(x, localmask)
        out_global = self.encode_global(x, globalmask, objmask)
        return out_local, out_global


class ClassAttributeDecoder(nn.Module):
    def __init__(self, c: EnvObjClass, dim_in, dim_hidden: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.c: Final = c
        self.fieldnames: Final = c.fieldnames('state')

        self.decoders = {
            fieldname: DistributionDecoder(
                dim_in, dim_hidden, c.field_vtypes[fieldname],
                device, dtype)
            for fieldname in self.fieldnames
        }
        for fieldname, module in self.decoders.items():
            self.add_module('%s_decoder' % fieldname, module)
    
    def decode_one(self, fieldname: str, x: torch.Tensor):
        '''
        x: (batch_size, n_obj_c, ? + n_field_c*dim_local)
        '''
        return self.decoders[fieldname].forward(x)

    def forward(self, x: torch.Tensor):
        if len(x) == 0:
            raise ValueError("The input dict inclues no attribute. "
                             f"Please check the definition of {self.c}")
        out = {fieldname: self.decode_one(fieldname, x[:, :, i])
               for i, fieldname in enumerate(self.fieldnames)}
        return out


class Inferer(nn.Module):
    def __init__(self, info: EnvInfo, 
            dim_variable_enc: int, dim_global: int, dim_local: int,
            dim_decoder_hidden: int, dim_reward_predictor_hidden: int,
            device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.info = info
        self.device = device
        self.dtype = dtype
        self.encoders = {
            c.name: ClassEncoder(c, dim_variable_enc, dim_local, dim_global, device, dtype)
            for c in info.classes}
        self.dim_global_enc = sum(encoder.dim_out_global
                                  for encoder in self.encoders.values())
        dim_in_decoders = {
            c.name: self.dim_global_enc + self.encoders[c.name].dim_out_local
            for c in info.classes}
        self.decoders = {
            c.name: ClassAttributeDecoder(
                c, dim_in_decoders[c.name], dim_decoder_hidden, device, dtype)
            for c in info.classes}
        
        for clsname, encoder in self.encoders.items():
            self.add_module(f"{clsname}_encoder", encoder)
        for clsname, decoder in self.decoders.items():
            self.add_module(f"{clsname}_decoder", decoder)

        dh = dim_reward_predictor_hidden
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.dim_global_enc, dh, dtype=dtype, device=device),
            nn.LeakyReLU(),
            nn.Linear(dh, dh, dtype=dtype, device=device),
            nn.LeakyReLU(),
            nn.Linear(dh, 1, device=device, dtype=dtype),
        )
    
    def get_global_encodings(self, variable_encodings: NamedTensors,
                             maskgen: MaskGenerator, objmasks: Optional[NamedTensors]):
        info = self.info
        batchsize = next(iter(variable_encodings.values())).shape[0]
        global_mask = maskgen.global_mask(batchsize)  # (batch_size, n_statefield, n_field)
        global_encs_list: List[torch.Tensor] = []
        for c in info.classes:
            if c.name in variable_encodings:
                x = variable_encodings[c.name]
                encoder = self.encoders[c.name]
                global_mask_c = global_mask[:, :, info.field_slice(c.name, 'all')]
                objmask = None if objmasks is None else objmasks[c.name]

                # attrmask: (batch_size, n_statefield, n_field_c)
                global_enc = encoder.encode_global(x, global_mask_c, objmask)
                global_encs_list.append(global_enc)
            else:
                encoder = self.encoders[c.name]
                global_enc = torch.zeros(
                    batchsize, info.n_field('state') + 1, encoder.dim_out_global,
                    dtype = self.dtype, device = self.device
                )
                global_encs_list.append(global_enc)
        global_encs = torch.cat(global_encs_list, dim=2)
        return global_encs

    def infer_one_class(self, clsname: str, global_encs: torch.Tensor,
            variable_encoding: torch.Tensor, maskgen: MaskGenerator):
        batchsize = global_encs.shape[0]
        local_mask = maskgen.local_mask(clsname, batchsize)
        encoder = self.encoders[clsname]

        # local_enc: (batch_size, n_obj_c, n_statefield_c, n_field_c*dim_local)
        local_enc = encoder.encode_local(variable_encoding, local_mask)

        # global_enc: (batch_size, n_statefield_c, ?)
        global_enc = global_encs[:, self.info.field_slice(clsname, 'state'), :]
        
        # global_enc: (batch_size, n_obj_c, n_statefield_c, ?)
        n_obj_c = local_enc.shape[1]
        global_enc = global_enc.unsqueeze(1).expand(-1, n_obj_c, -1, -1)

        # x: (batch_size, n_obj_c, n_statefield_c, ? + n_field_c*dim_local)
        x = torch.cat((local_enc, global_enc), dim=3)
        return self.decoders[clsname].forward(x)
    
    def infer_reward(self, global_encs: torch.Tensor):
        x = global_encs[:, -1, :]
        y: torch.Tensor = self.reward_predictor(x)
        return y.squeeze(dim=1)

    def forward(self, variable_encodings: NamedTensors,
                maskgen: MaskGenerator, objmasks: Optional[NamedTensors] = None):
        
        global_encs = self.get_global_encodings(variable_encodings, maskgen, objmasks)
        out: ObjectDistributions = {}
        for clsname, x in variable_encodings.items():            
            out[clsname] = self.infer_one_class(clsname, global_encs, x, maskgen)
        reward = self.infer_reward(global_encs)
        return out, reward


class EnvModel(nn.Module):

    class Args(utils.Struct):
        def __init__(self) -> None:
            self.dim_variable_encoding: int = 8
            self.dim_variable_encoder_hidden: int = 8
            self.variable_encoder_norm_momentum: float = 0.01
            self.dim_local_hidden: int = 8
            self.dim_global_hidden: int = 8
            self.dim_decoder_hidden: int = 32
            self.dim_reward_predictor_hidden: int = 32

    def __init__(self, envinfo: EnvInfo, args: 'EnvModel.Args',
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.variable_encoder = VariableEncoder(
            envinfo,
            args.dim_variable_encoder_hidden,
            args.dim_variable_encoding,
            args.variable_encoder_norm_momentum,
            device, dtype)

        self.inferer = Inferer(envinfo, args.dim_variable_encoding,
            args.dim_local_hidden, args.dim_global_hidden,
            args.dim_decoder_hidden, args.dim_reward_predictor_hidden,
            device, dtype)

        for p in self.parameters():
            if p.ndim >= 2:
                nn.init.xavier_normal_(p)
            else:
                nn.init.normal_(p)

    def forward(self, raw_attributes: ObjectTensors,
                attr_mask_generator: MaskGenerator, 
                object_mask: Optional[NamedTensors] = None):
        
        encodings = self.variable_encoder.forward(raw_attributes)
        state, reward = self.inferer.forward(
            encodings, attr_mask_generator, object_mask)
        return state, reward
