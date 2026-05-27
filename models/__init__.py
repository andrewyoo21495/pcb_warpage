"""Model factory for PCB Warpage generative models."""


def build_model(config: dict):
    """Instantiate the model specified by config['model_type'].

    Args:
        config: dict from load_config().

    Returns:
        nn.Module — CVAE, ConditionalDDPM, LatentDiffusionModel, or LatentFlowMatching.
    """
    model_type = str(config.get('model_type', 'cvae')).lower()

    if model_type == 'cvae':
        from models.cvae import CVAE
        return CVAE(config)
    elif model_type == 'ddpm':
        from models.ddpm import ConditionalDDPM
        return ConditionalDDPM(config)
    elif model_type == 'ldm':
        from models.ldm import LatentDiffusionModel
        return LatentDiffusionModel(config)
    elif model_type == 'lfm':
        from models.lfm import LatentFlowMatching
        return LatentFlowMatching(config)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            "Choose 'cvae', 'ddpm', 'ldm', or 'lfm'."
        )
