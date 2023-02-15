import os
import logging

import wblog

def enable_debug(*, validation=True, api_dump=True):
    """
    Enable debug message
    """
    layers = []
    if validation:
        layers.append("VK_LAYER_KHRONOS_validation")
    if api_dump:
        layers.append("VK_LAYER_LUNARG_api_dump")

    if len(layers) > 0:
        os.environ["VK_INSTANCE_LAYERS"] = ":".join(layers)

    wblog.start_logging("vulkpy", level=logging.DEBUG)


def getShader(name: str):
    """
    Get Shader Path

    Parameters
    ----------
    name : str
        SPIR-V (.spv) name
    """
    return os.path.join(os.path.dirname(__file__), "shader", name)
