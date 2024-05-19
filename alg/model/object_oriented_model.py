from typing import Final, Dict, Optional, Iterable, Tuple, List, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
import numpy as np

from core import VType, DType, EnvObjClass, ObjectOrientedEnv, EnvInfo
from utils.typings import ObjectTensors, NamedTensors, ObjectDistributions, TransitionModel
import utils
import alg.functional as F
from .mask_generator import MaskGenerator

from .modules import MultiLinear, HeterogenousLinear, attention
from .base import RLModel


class DistributionDecoder(nn.Module):
    def __init__(self, dim_in: int, vtype: VType,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()
        self._vtype = vtype
        self._ptype = vtype.ptype
        self.sub_decoders = {
            key: nn.Linear(dim_in, dim_param, device=device, dtype=dtype)
            for key, dim_param in self._ptype.param_sizes.items()}
        for param, decoder in self.sub_decoders.items():
            self.add_module(f"{param} decoder", decoder)

    def forward(self, x: Tensor):
        params = {k: decoder(x) for k, decoder in self.sub_decoders.items()}
        out = self._ptype(**params)
        return out


class ClassAttributeEncoder(nn.Module):
    def __init__(self, c: EnvObjClass, dim_hidden: int, dim_encoding: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.c: Final = c
        self.fieldnames: Final = c.fieldnames()
        self.n_field = c.n_field()

        self.inputer_ = HeterogenousLinear(
            [c.v(a).size for a in self.fieldnames], dim_hidden, dtype, device)
        self.encoder = nn.Sequential(
            nn.LeakyReLU(),
            MultiLinear.auto([self.n_field], dim_hidden, dim_encoding, dtype, device),
            nn.ReLU(),
        )
        

    def forward(self, raws: NamedTensors) -> Tensor:
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
        xs = [self.c.v(fieldname).raw2input(raws[fieldname])
              for fieldname in self.fieldnames]
        x = self.inputer_.forward(xs)  # (batchsize, n_obj, n_field, dim_h)
        x: Tensor = self.encoder(x)
        return x


class VariableEncoder(nn.Module):

    def __init__(self, envinfo: EnvInfo, dim_hidden: int, dim_out: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.encoders = {
            c.name: ClassAttributeEncoder(
                c, dim_hidden, dim_out,
                device, dtype)
            for c in envinfo.classes
        }
        for cls_name, module in self.encoders.items():
            self.add_module('%s_encoder' % cls_name, module)

    def forward(self, raw_attributes: ObjectTensors) -> NamedTensors:
        return {cls_name: self.encoders[cls_name].forward(x)
                for cls_name, x in raw_attributes.items()}


class ClassEncoder(nn.Module):
    def __init__(self, info: EnvInfo, c: EnvObjClass,
                 dim_in: int, dim_local: int, dim_global: int,
                 dim_id: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        dim_in = dim_in * c.n_field() + dim_id
        self.f_local = MultiLinear.auto([c.n_field('state')], dim_in, dim_local, dtype, device)
        self.f_global = MultiLinear.auto([info.n_field('state')], dim_in, dim_global, dtype, device)

        self.__include_id = (dim_id > 0)
        if self.__include_id:
            if dim_id % 2 != 0:
                raise ValueError
            self.idencoder = nn.GRU(0, dim_id // 2, batch_first=True, bidirectional=True,
                                    device=device, dtype=dtype)
            _h0 = torch.zeros(2, 1, dim_id // 2, device=device, dtype=dtype)
            self.idencoder_h0 = nn.Parameter(_h0)

    def __encode_id(self, x: Tensor):
        batchsize, n_obj = x.shape[:2]
        input_ = torch.zeros(1, n_obj, 0, device=x.device, dtype=x.dtype)
        enc, _ = self.idencoder(input_, self.idencoder_h0)
        enc: Tensor  # (1, n_obj, dim_id)
        enc = enc.unsqueeze(2).expand(*x.shape[:3], -1)
        return torch.cat((x, enc), dim=-1)

    def encode_local(self, x: Tensor, attrmask: Tensor):
        '''
        Args:
            x: (batch_size, n_obj_c, n_field_c, dim_in)
            attrmask: (batch_size, n_statefield_c, n_field_c)
        Returns:
            Tensor: (batchsize, n_obj_c, n_statefield, dim_local)
        '''
        # (batch_size, n_obj_c, n_statefield_c, n_field_c, dim_in)
        x = F.masked_retain(x.unsqueeze(dim=2),
                            attrmask.unsqueeze(dim=-1).unsqueeze(1))
        # (batch_size, n_obj_c, n_statefield_c, n_field_c*dim_in)
        x = x.flatten(start_dim=-2)

        if self.__include_id:
            x = self.__encode_id(x)

        # (batch_size, n_obj_c, n_statefield_c, dim_local)
        # x = nn.functional.leaky_relu(self.f_local.forward(x))
        x = self.f_local.forward(x)
        return x

    def encode_global(self, x: Tensor, attrmask: Tensor):
        '''
        Args:
            x: (batch_size, n_obj_c, n_field_c, dim_in)
            attrmask: (batch_size, n_statefield, n_field_c)
        Returns:
            Tensor: (batchsize, n_obj_c, n_statefield, dim_global)
        '''

        # (batch_size, n_obj_c, n_statefield, n_field_c, dim_in)
        x = F.masked_retain(x.unsqueeze(dim=2),
                            attrmask.unsqueeze(dim=-1).unsqueeze(1))

        # (batch_size, n_obj_c, n_statefield, n_field_c*dim_global)
        x = x.flatten(start_dim=-2)

        if self.__include_id:
            x = self.__encode_id(x)

        # (batch_size, n_obj_c, n_statefield, dim_global)
        # x = nn.functional.leaky_relu(self.f_global.forward(x))
        x = self.f_global.forward(x)
        return x

    def forward(self, x: Tensor, localmask: Tensor, globalmask: Tensor):
        out_local = self.encode_local(x, localmask)  # (batch_size, n_obj_c, n_statefield_c, dim_local)
        out_global = self.encode_global(x, globalmask)  # (batch_size, n_obj_c, n_statefield, dim_global)
        return out_local, out_global


class AttributeDecoder(nn.Module):
    def __init__(self, c: EnvObjClass, dim_in: int, dim_hidden: int,
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.c: Final = c
        self.fieldnames: Final = c.fieldnames('state')

        n_field_s = c.n_field('state')
        self.f = nn.Sequential(
            MultiLinear.auto([n_field_s], dim_in, dim_hidden, dtype, device),
            nn.LeakyReLU(),
            MultiLinear.auto([n_field_s], dim_hidden, dim_hidden, dtype, device),
            nn.LeakyReLU(),
        )
        self.decoders = {
            fieldname: DistributionDecoder(
                dim_hidden, c.field_vtypes[fieldname],
                device, dtype)
            for fieldname in self.fieldnames
        }
        for fieldname, module in self.decoders.items():
            self.add_module('%s_decoder' % fieldname, module)

    def forward(self, x: Tensor):
        '''
        x: (batchsize, n_obj_c, n_field_state_c, dim_in)
        '''
        if len(x) == 0:
            raise ValueError("The input dict inclues no attribute. "
                             f"Please check the definition of {self.c}")
        x = self.f(x)
        out = {fieldname: self.decoders[fieldname].forward(x[:, :, i])
               for i, fieldname in enumerate(self.fieldnames)}
        return out


class ClassInferer(nn.Module):
    def __init__(self, c: EnvObjClass,
                 dim_global: int, dim_local: int, dim_k: int, dim_v: int,
                 dim_decoder_hidden: int, n_head: Optional[int],
                 device: torch.device, dtype: torch.dtype):
        super().__init__()

        self.fk = MultiLinear.auto([c.n_field('state')], dim_global, dim_k, dtype, device)
        self.fq = MultiLinear.auto([c.n_field('state')], dim_local, dim_k, dtype, device)
        self.fv = MultiLinear.auto([c.n_field('state')], dim_global, dim_v, dtype, device)
        self.decoder = AttributeDecoder(
            c, dim_v + dim_local, dim_decoder_hidden, device, dtype)
        self.n_head = n_head

    def forward(self, local_enc: Tensor, global_enc: Tensor,
                attnmask: Optional[Tensor] = None):
        """
        Args:
            local_enc (Tensor): (batchsize, n_obj_c, n_field_state_c, dim_local)
            global_enc (Tensor): (batchsize, n_obj, n_field_state_c, dim_local)
            attnmask (Tensor, optional): (batchsize, n_obj_c, n_obj). Defaults to None.

        Returns:
            Dict[str, Distribution]: distributions of attribtues
            Tensor (Optional): the encoding tensor for predicting reward, if `encode_reward` is True.
        """

        k = self.fk.forward(global_enc)  # (batchsize, n_obj, n_field_state_c, dim_k)
        v = self.fv.forward(global_enc)  # (batchsize, n_obj, n_field_state_c, dim_v)
        q = self.fq.forward(local_enc)  # (batchsize, n_obj_c, n_field_state_c, dim_k)

        k = k.transpose(1, 2)  # (batchsize, n_field_state_c, n_obj, dim_k)
        v = v.transpose(1, 2)  # (batchsize, n_field_state_c, n_obj, dim_v)
        q = q.transpose(1, 2)  # (batchsize, n_field_state_c, n_obj_c, dim_k)
        
        if attnmask is not None:
            attnmask = attnmask.unsqueeze(1)

        v_ = attention(q, k, v, self.n_head, attnmask)  # (batchsize, n_field_state_c, n_obj_c, dim_v)
        v_ = v_.transpose(1, 2)  # (batchsize, n_obj_c, n_field_state_c, dim_v)
        x = torch.cat((local_enc, v_), dim=-1)  # (batchsize, n_obj_c, n_field_state_c, dim_v + dim_local)

        distr = self.decoder.forward(x)
        return distr


class Inferer(nn.Module):
    def __init__(self, info: EnvInfo, 
            dim_variable_enc: int, dim_global: int, dim_local: int,
            dim_k: int, dim_v: int, dim_decoder_hidden: int,
            n_head: Optional[int], dim_id: int, 
            device: torch.device, dtype: torch.dtype) -> None:
        
        super().__init__()
        self.info = info
        self.device = device
        self.dtype = dtype
        
        self.encoders = {
            c.name: ClassEncoder(info, c, dim_variable_enc, dim_local, dim_global,
                                 dim_id, device, dtype)
            for c in info.classes}
    
        self.inferers = {
            c.name: ClassInferer(c, dim_global, dim_local, dim_k, dim_v,
                                 dim_decoder_hidden, n_head, device, dtype)
            for c in info.classes}
        
        for clsname, encoder in self.encoders.items():
            self.add_module(f"{clsname}_encoder", encoder)
        for clsname, inferer in self.inferers.items():
            self.add_module(f"{clsname}_inferer", inferer)
    
    def get_global_encodings(self, variable_encodings: NamedTensors, maskgen: MaskGenerator):
        '''
        returns:
            Tensor: (batchsize, n_obj, n_field_state, dim_global)
            Dict[str, Tensor]: {classname: (batchsize, n_obj_c, n_field_state_c, dim_local)}
        '''
        info = self.info
        batchsize = next(iter(variable_encodings.values())).shape[0]
        global_mask = maskgen.global_mask(batchsize)  # (batch_size, n_statefield, n_field)
        global_encs_list: List[Tensor] = []
        local_encs_dict: Dict[str, Tensor] = {}
        for clsname, x in variable_encodings.items():
            encoder = self.encoders[clsname]
            local_mask_c = maskgen.local_mask(clsname, batchsize)
            global_mask_c = global_mask[:, :, info.field_slice(clsname, 'all')]
            global_enc = encoder.encode_global(x, global_mask_c)
            global_encs_list.append(global_enc)
        global_encs = torch.cat(global_encs_list, dim=1)
        return global_encs

    def get_attn_masks(self, variable_encodings: NamedTensors,
                       objmasks: Optional[NamedTensors] = None):

        ranges: Dict[str, Tuple[int, int]] = {}
        temp = []
        n_obj = 0
        for clsname, x in variable_encodings.items():
            b, n_obj_c, n_field, _ = x.shape
            ranges[clsname] = (n_obj, n_obj + n_obj_c)
            n_obj += n_obj_c
            if objmasks is None:
                m = torch.ones(b, n_obj_c, dtype=torch.bool, device=self.device)
            else:
                m = objmasks[clsname]
            temp.append(m)
        
        objmask = torch.cat(temp, dim=1)  # (batchsize, n_obj)

        attnmasks: NamedTensors = {}
        for clsname, x in variable_encodings.items():
            b, n_obj_c, n_field, _ = x.shape
            m = objmask.unsqueeze(1).repeat(1, n_obj_c, 1)  # (batchsize, n_obj_c, n_obj)
            i, j = ranges[clsname]
            if objmasks is not None:
                m = F.masked_retain(m, objmasks[clsname].unsqueeze(-1), False)
            m[:, range(n_obj_c), range(i, j)] = False  # no attention to itself
            attnmasks[clsname] = m
        
        return attnmasks

    def infer_one_class(self, clsname: str, global_encs: Tensor,
            variable_encoding: Tensor, maskgen: MaskGenerator,
            attnmask: Optional[Tensor] = None):
        batchsize = global_encs.shape[0]
        local_mask = maskgen.local_mask(clsname, batchsize)
        encoder = self.encoders[clsname]

        # local_enc: (batch_size, n_obj_c, n_statefield_c, n_field_c*dim_local)
        local_enc = encoder.encode_local(variable_encoding, local_mask)

        # global_enc: (batch_size, n_obj, n_statefield_c, ?)
        global_enc = global_encs[:, :, self.info.field_slice(clsname, 'state'), :]

        return self.inferers[clsname].forward(local_enc, global_enc, attnmask)

    def forward(self, variable_encodings: NamedTensors,
                maskgen: MaskGenerator, objmasks: Optional[NamedTensors] = None):
        attnmasks = self.get_attn_masks(variable_encodings, objmasks)
        global_encs = self.get_global_encodings(variable_encodings, maskgen)
        out: ObjectDistributions = {}
        for clsname, x in variable_encodings.items():
            attnmask = attnmasks[clsname]
            out[clsname] = self.infer_one_class(clsname, global_encs, x, maskgen, attnmask)
        return out


class OOCModel(RLModel):
    '''Object-Oriented Causal Model'''

    class Args(utils.Struct):
        def __init__(self,
                dim_variable_encoding: int = 16,
                dim_variable_encoder_hidden: int = 16,
                dim_local: int = 32,
                dim_global: int = 32,
                dim_id: int = 0,
                dim_decoder_hidden: int = 32,
                dim_k = 32,
                dim_v = 32,
                n_attn_head: Optional[int] = 2
        ):
            self.dim_variable_encoding = dim_variable_encoding
            self.dim_variable_encoder_hidden = dim_variable_encoder_hidden
            self.dim_local = dim_local
            self.dim_global = dim_global
            self.dim_id = dim_id
            self.dim_decoder_hidden = dim_decoder_hidden
            self.dim_k = dim_k
            self.dim_v = dim_v
            self.n_attn_head= n_attn_head

    def __init__(self, env, args, device, dtype):
        super().__init__(env, args, device, dtype)

        envinfo = env.info

        self.variable_encoder = VariableEncoder(
            envinfo,
            args.dim_variable_encoder_hidden,
            args.dim_variable_encoding,
            device, dtype)

        self.inferer = Inferer(envinfo,
            args.dim_variable_encoding,
            args.dim_global, args.dim_local,
            args.dim_k, args.dim_v, args.dim_decoder_hidden,
            args.n_attn_head, args.dim_id,
            device, dtype)

    def forward(self, raw_attributes: ObjectTensors,
                attr_mask_generator: MaskGenerator, 
                object_mask: Optional[NamedTensors] = None):
        encodings = self.variable_encoder.forward(raw_attributes)
        state = self.inferer.forward(encodings, attr_mask_generator, object_mask)
        return state

    def make_transition_model(self, maskgen: MaskGenerator) -> TransitionModel:
        def f(raw_attributes: ObjectTensors, object_mask: Optional[NamedTensors] = None):
            return self.forward(raw_attributes, maskgen, object_mask)
        return f



class OOCHModel(RLModel):
    '''Object-Oriented Causal Model, homogeneous'''

    class Args(utils.Struct):
        def __init__(self,
                dim_variable_encoding: int = 16,
                dim_variable_encoder_hidden: int = 16,
                dim_local: int = 32,
                dim_global: int = 32,
                dim_id: int = 0,
                dim_decoder_hidden: int = 32,
                dim_k = 32,
                dim_v = 32,
                n_attn_head: Optional[int] = 2
        ):
            self.dim_variable_encoding = dim_variable_encoding
            self.dim_variable_encoder_hidden = dim_variable_encoder_hidden
            self.dim_local = dim_local
            self.dim_global = dim_global
            self.dim_id = dim_id
            self.dim_decoder_hidden = dim_decoder_hidden
            self.dim_k = dim_k
            self.dim_v = dim_v
            self.n_attn_head= n_attn_head

    def __init__(self, env, args, device, dtype):
        super().__init__(env, args, device, dtype)

        envinfo = env.info

        self.variable_encoder = VariableEncoder(
            envinfo,
            args.dim_variable_encoder_hidden,
            args.dim_variable_encoding,
            device, dtype)

        self.inferer = Inferer(envinfo,
            args.dim_variable_encoding,
            args.dim_global, args.dim_local,
            args.dim_k, args.dim_v, args.dim_decoder_hidden,
            args.n_attn_head, args.dim_id,
            device, dtype)

    def forward(self, raw_attributes: ObjectTensors,
                attr_mask_generator: MaskGenerator, 
                object_mask: Optional[NamedTensors] = None):
        encodings = self.variable_encoder.forward(raw_attributes)
        state = self.inferer.forward(encodings, attr_mask_generator, object_mask)
        return state

    def make_transition_model(self, maskgen: MaskGenerator) -> TransitionModel:
        def f(raw_attributes: ObjectTensors, object_mask: Optional[NamedTensors] = None):
            return self.forward(raw_attributes, maskgen, object_mask)
        return f
