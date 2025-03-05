from .cora import load_cora
from .citeseer import load_citeseer
from .mutag import load_mutag
from .fb15k import load_fb15k

__all__ = [
    'load_cora',
    'load_citeseer',
    'load_mutag',
    'load_fb15k'
]