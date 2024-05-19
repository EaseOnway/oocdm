from typing import Dict, Callable, TypeVar
import torch
from torch.distributions import Distribution, kl_divergence as _kl

from core import EnvInfo
from utils.typings import NamedTensors, NamedDistributions, \
    ObjectTensors, ObjectArrays, ObjectDistributions, ShapeLike, \
    ObjectValues
import utils



_Tin = TypeVar("_Tin")
_Tout = TypeVar("_Tout")


def init_params(model: torch.nn.Module):
    for p in model.parameters():
        if p.ndim >= 2:
            torch.nn.init.xavier_normal_(p)
        else:
            torch.nn.init.normal_(p)    


def apply(inputs: ObjectValues[_Tin],
          func: Callable[[str, str, _Tin], _Tout]) -> ObjectValues[_Tout]:
    return {clsname: {fieldname: func(clsname, fieldname, x)
                      for fieldname, x in c_inputs.items()}
     for clsname, c_inputs in inputs.items()}


def sample(distributions: ObjectDistributions, batch_shape: ShapeLike = ()):
    size = torch.Size(utils.Shaping.as_shape(batch_shape))
    return {clsname: {fieldname: d.sample(size) for fieldname, d in temp.items()}
            for clsname, temp in distributions.items()}

def mode(distributions: ObjectDistributions):
    return {clsname: {fieldname: d.mode for fieldname, d in temp.items()}
            for clsname, temp in distributions.items()}

def logprob(distributions: ObjectDistributions,
            label: ObjectTensors) -> Dict[str, NamedTensors]:
    return {clsname: {
                fieldname: torch.sum(d.log_prob(label[clsname][fieldname]), dim=2)
                for fieldname, d in temp.items()}
            for clsname, temp in distributions.items()}

def sum_logprob_by_class(logprob: ObjectTensors, objmask: NamedTensors):
    logp_list = []
    for clsname, temp in logprob.items():
        m = objmask[clsname]  # Tensor(batchsize, n_obj)
        cls_logp = torch.sum(
            torch.stack([logp for fieldname, logp in temp.items()], dim=2),
            dim=2)  # (batchsize, n_obj)
        cls_logp = torch.mean(torch.masked_select(cls_logp, m))  # scalar
        logp_list.append(cls_logp)

    return torch.sum(torch.stack(logp_list))


def sum_logprob_by_variable(logprob: ObjectTensors, objmask: NamedTensors):
    logp_list = []
    for clsname, temp in logprob.items():
        m = objmask[clsname]  # Tensor(batchsize, n_obj)
        cls_logp = torch.stack([logp for fieldname, logp in temp.items()], dim=2)  # (batchsize, n_obj, n_field)
        logp_list.append(torch.masked_select(cls_logp, m.unsqueeze(-1)))
    return torch.mean(torch.cat(logp_list))


def kl(ps: ObjectDistributions, qs: ObjectDistributions):
    return {clsname: {fieldname: _kl(p, qs[clsname][fieldname])
                      for fieldname, p in temp.items()}
            for clsname, temp in ps.items()}

def entropy(distributions: ObjectDistributions):
    return {clsname: {fieldname: d.entropy()
                      for fieldname, d in temp.items()}
            for clsname, temp in distributions.items()}

def raws2labels(envinfo: EnvInfo, raw_attributes: ObjectTensors):
    return {
        clsname: {
            fieldname: envinfo.v(clsname, fieldname).raw2label(x)
            for fieldname, x in temp.items()
        }
        for clsname, temp in raw_attributes.items()
    }


def labels2raws(envinfo: EnvInfo, labels: ObjectTensors) -> ObjectTensors:
    return {
        clsname: {
            fieldname: envinfo.v(clsname, fieldname).label2raw(x)
            for fieldname, x in temp.items()
        }
        for clsname, temp in labels.items()
    }


def sample_raw(envinfo: EnvInfo, distributions: ObjectDistributions,
               batch_shape: ShapeLike = ()) -> ObjectTensors:
    size = torch.Size(utils.Shaping.as_shape(batch_shape))
    return {
        clsname: {
            fieldname: envinfo.v(clsname, fieldname).label2raw(d.sample(size))
            for fieldname, d in temp.items()
        }
        for clsname, temp in distributions.items()
    }


def mode_raw(envinfo: EnvInfo, distributions: ObjectDistributions) -> ObjectTensors:
    return {
        clsname: {
            fieldname: envinfo.v(clsname, fieldname).label2raw(d.mode)
            for fieldname, d in temp.items()
        }
        for clsname, temp in distributions.items()
    }


def as_raw_tensors(envinfo: EnvInfo, raw_attributes: ObjectArrays,
                   device: torch.device) -> ObjectTensors:
    return {
        clsname: {
            fieldname: envinfo.v(clsname, fieldname).tensor(x, device)
            for fieldname, x in temp.items()
        }
        for clsname, temp in raw_attributes.items()
    }


def as_arrays(raw_attributes: ObjectTensors) -> ObjectArrays:
    return {
        clsname: {
            fieldname: x.detach().to(device='cpu').numpy()
            for fieldname, x in temp.items()
        }
        for clsname, temp in raw_attributes.items()
    }


def masked_retain(x: torch.Tensor, m: torch.Tensor, value=0):
    return torch.masked_fill(x, m.logical_not(), value)
