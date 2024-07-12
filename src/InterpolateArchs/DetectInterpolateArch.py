"""
https://github.com/chaiNNer-org/spandrel/blob/main/libs/spandrel/spandrel/util/__init__.py#L13
"""

from __future__ import annotations

import functools
import inspect
import math
from typing import Any, Literal, Mapping, Protocol, TypeVar
import torch
from .RIFE import rife46IFNET, rife413IFNET

rife_archs = [rife46IFNET, rife413IFNET]

class loadModel:
    """
    Pass in a state dict of a rife model, and will automatically load the correct archetecture
    """
    def __init__(self, state_dict: dict):
        self.state_dict = state_dict
        self.detect()
    def detect(self):
        for arch in rife_archs:
            if (list(self.state_dict.keys()) == arch.keys()):
                return arch
