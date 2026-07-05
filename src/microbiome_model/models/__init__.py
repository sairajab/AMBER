import os
import torch

from microbiome_model.models.orig import (
    BasicRegressor,
    AMBERRegressor,
    BasicRegressorwithUnifrac,
)
from microbiome_model.models.zoo import (
    BasicRegressorGRL,
    CLR,
    ClusteredRegressor,
    GeneralizedRegressor,
    BasicRegressorBest,
    BasicRegressor1, 
    BasicRegressor2,
    BasicRegressor3,
    BasicRegressor4,
    BasicRegressor5,
    BasicRegressor6,
    BasicRegressorAttn
)


def build_model(cfg: dict, device: str = "cpu"):
    """Construct and return a model from a config dict.

    The config must include ``model_name``.  All other keys are forwarded
    to the constructor of the chosen class.  Supported names:

    - ``BasicRegressor``
    - ``BasicRegressorGRL``
    - ``BasicRegressorNew``
    - ``GeneralizedRegressor``
    - ``CLR``
    - ``ClusteredRegressor``
    - ``AttnRegressor``

    For ``BasicRegressorNew`` the config may include
    ``pretrained_weights`` (absolute path to a ``.pt`` file).
    """
    name = cfg["model_name"]

    if name == "BasicRegressor":
        return BasicRegressor(
            input_dim=cfg.get("input_dim", 768),
            hidden_dim=cfg.get("hidden_dim", 1024),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            pe=cfg.get("pe", False),
            grl =cfg.get("grl", False),
            unique_donors_train=cfg.get("num_donors", 0),
            clr =cfg.get("clr", False),
            metadata_cardinalities=cfg.get("metadata_cardinalities", None),
        )
    elif name == "AMBERRegressor":
        return AMBERRegressor(
            input_dim=cfg.get("input_dim", 768),
            hidden_dim=cfg.get("hidden_dim", 1024),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            pe=cfg.get("pe", False),
            metadata_cardinalities=cfg.get("metadata_cardinalities", None),
        )

    elif name == "BasicRegressorGRL":
        return BasicRegressorGRL(
            input_dim=cfg.get("input_dim", 768),
            hidden_dim=cfg.get("hidden_dim", 1024),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            pe=cfg.get("pe", False),
            metadata_cardinalities=cfg.get("metadata_cardinalities", None),
        )

    elif name == "BasicRegressor1":
        return BasicRegressor1(
            input_dim=cfg.get("input_dim", 768),
            hidden_dim=cfg.get("hidden_dim", 1024),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            pe=cfg.get("pe", False),
            metadata_cardinalities=cfg.get("metadata_cardinalities", None),
        )
    
    elif name == "BasicRegressor2":
        return BasicRegressor2(
            input_dim=cfg.get("input_dim", 768),
            hidden_dim=cfg.get("hidden_dim", 1024),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            pe=cfg.get("pe", False),
            metadata_cardinalities=cfg.get("metadata_cardinalities", None),
        )
    
    elif name == "BasicRegressor3":
        return BasicRegressor3(
            input_dim=cfg.get("input_dim", 768),
            hidden_dim=cfg.get("hidden_dim", 1024),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            pe=cfg.get("pe", False),
            metadata_cardinalities=cfg.get("metadata_cardinalities", None),
        )

    elif name == "BasicRegressorAttn":
        return BasicRegressorAttn(
            input_dim=cfg.get("input_dim", 768),
            hidden_dim=cfg.get("hidden_dim", 1024),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            pe=cfg.get("pe", False),
            metadata_cardinalities=cfg.get("metadata_cardinalities", None),
        )
    elif name == "BasicRegressor4":
        return BasicRegressor4(
            input_dim=cfg.get("input_dim", 768),
            hidden_dim=cfg.get("hidden_dim", 1024),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            pe=cfg.get("pe", False),
            metadata_cardinalities=cfg.get("metadata_cardinalities", None),
        )
    elif name == "BasicRegressor5":
        return BasicRegressor5(
            input_dim=cfg.get("input_dim", 768),
            hidden_dim=cfg.get("hidden_dim", 1024),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            pe=cfg.get("pe", False),
            metadata_cardinalities=cfg.get("metadata_cardinalities", None),
            grl=True,
            unique_donors_train=cfg.get("num_donors", 74),
        )
    elif name == "BasicRegressor6":
        return BasicRegressor6(
            input_dim=cfg.get("input_dim", 768),
            hidden_dim=cfg.get("hidden_dim", 1024),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            pe=cfg.get("pe", False),
            metadata_cardinalities=cfg.get("metadata_cardinalities", None),
        )
    elif name == "BasicRegressorNew":
        from microbiome_model.training.pre_training_masked import MaskedAbundancePretraining
        base = MaskedAbundancePretraining(input_dim=768)
        weights_path = cfg.get("pretrained_weights", "")
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=device)
            base.load_state_dict(state, strict=False)
            print(f"Loaded pretrained weights from {weights_path}")
        else:
            print("pretrained_weights not found; BasicRegressorNew initialised from scratch.")
        return BasicRegressorNew(
            basemodel=base,
            input_dim=cfg.get("input_dim", 768),
            hidden_dim=cfg.get("hidden_dim", 1024),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            pe=cfg.get("pe", False),
            bins=cfg.get("bins", False),
        )

    elif name == "GeneralizedRegressor":
        return GeneralizedRegressor(
            input_dim=cfg.get("input_dim", 768),
            hidden_dim=cfg.get("hidden_dim", 512),
            num_heads=cfg.get("num_heads", 8),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.3),
        )

    elif name == "CLR":
        return CLR(
            proj_dim=cfg.get("proj_dim", 64),
            hidden_dim=cfg.get("hidden_dim", 256),
            dropout=cfg.get("dropout", 0.3),
        )

    elif name == "ClusteredRegressor":
        return ClusteredRegressor(
            input_dim=cfg.get("input_dim", 256),
            hidden_dim=cfg.get("hidden_dim", 512),
            num_clusters=cfg.get("num_clusters", 32),
            num_heads=cfg.get("num_heads", 8),
            dropout=cfg.get("dropout", 0.2),
        )

    elif name == "AttnRegressor":
        return AttnRegressor(
            input_dim=cfg.get("input_dim", 512),
            hidden_dim=cfg.get("hidden_dim", 512),
            num_heads=cfg.get("num_heads", 8),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
        )
    elif name == "BasicRegressorBest":
        return BasicRegressorBest(
            input_dim=cfg.get("input_dim", 512),
            hidden_dim=cfg.get("hidden_dim", 512),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            pe=cfg.get("pe", False),
            )

    else:
        raise ValueError(
            f"Unknown model_name: {name!r}. "
            "Supported: BasicRegressor, BasicRegressorGRL, BasicRegressorNew, BasicRegressor1, "
            "GeneralizedRegressor, CLR, ClusteredRegressor, AttnRegressor."
        )
