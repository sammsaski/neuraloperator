from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.si_block import SIBlocks
from .base_model import BaseModel

class SINO(BaseModel, name='SINO'):
    """Spline-Integral Neural Operator. The SINO is useful for solving IVPs.

    Parameters
    ----------


    Other parameters
    ----------------


    Examples
    --------


    References
    ----------

    """

    def __init__(
		self,
    ):
		super().__init__()
