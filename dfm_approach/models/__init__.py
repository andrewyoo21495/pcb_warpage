#!/usr/bin/env python3
"""DF²M model factory and unified interface.

Provides build_dfm_models(config) to construct all three modules:
    Module A: FNOMeanPredictor
    Module B-1: ConditionEncoder + ResidualCAE
    Module B-2: OTCFM (velocity network)
"""

from .fno_mean_predictor import FNOMeanPredictor
from .condition_encoder import ConditionEncoder
from .residual_cae import ResidualCAE
from .velocity_net import VelocityNet
from .ot_cfm import OTCFM


def build_dfm_models(config: dict) -> dict:
    """Build all DF²M model components from config.

    Returns:
        dict with keys:
            'fno':     FNOMeanPredictor  (Module A)
            'cond_enc': ConditionEncoder (shared condition encoder)
            'cae':     ResidualCAE       (Module B-1)
            'cfm':     OTCFM             (Module B-2)
    """
    return {
        'fno': FNOMeanPredictor(config),
        'cond_enc': ConditionEncoder(config),
        'cae': ResidualCAE(config),
        'cfm': OTCFM(config),
    }


__all__ = [
    'FNOMeanPredictor',
    'ConditionEncoder',
    'ResidualCAE',
    'VelocityNet',
    'OTCFM',
    'build_dfm_models',
]
