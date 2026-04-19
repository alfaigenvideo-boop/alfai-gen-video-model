import logging
from typing import Any
from .base_flow import BaseFlowEstimator
from .raft_flow import RAFTFlow
from .gmflow_flow import GMFlow

logger = logging.getLogger(__name__)

def build_flow(name: str = "raft", **kwargs: Any) -> BaseFlowEstimator:
    """
    Factory function to instantiate the requested optical flow model.
    Ensures the returned model adheres to the BaseFlowEstimator interface.

    Args:
        name (str): The name of the flow model to instantiate. Supported: 'raft', 'gmflow'.
        **kwargs: Additional configuration parameters passed to the model's constructor.

    Returns:
        BaseFlowEstimator: An instantiated optical flow model.

    Raises:
        ValueError: If the requested flow model name is not supported.
    """
    name = name.lower().strip()

    if name == "raft":
        logger.info("Instantiating RAFT optical flow model.")
        return RAFTFlow(**kwargs)

    elif name == "gmflow":
        logger.info("Instantiating GMFlow optical flow model.")
        return GMFlow(**kwargs)

    else:
        supported_models = ["raft", "gmflow"]
        error_msg = f"Unknown flow model requested: '{name}'. Supported models are: {supported_models}"
        logger.error(error_msg)
        raise ValueError(error_msg)