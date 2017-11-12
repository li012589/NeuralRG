from .model.template import RealNVPtemplate, PriorTemplate
from .model.realnvp import Gaussian, MLP, RealNVP, CNN
from .model.parallel import parallelize

__all__ = ['RealNVPtemplate', 'PriorTemplate',
           'Gaussian', 'MLP', 'RealNVP', 'CNN','parallelize']
