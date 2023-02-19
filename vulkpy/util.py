"""
Utility Module (:mod:`vulkpy.util`)
===================================


Examples
--------
>>> from vulkpy.util import enable_debug
>>> enable_debug(api_dump=False)
"""

import os
import logging

import wblog
logger = wblog.getLogger()


def enable_debug(*, validation: bool = True, api_dump: bool = True):
    """
    Enable debug message

    Parameters
    ----------
    validation : bool, optional
        If ``True`` (default), enable vulkan validation.
    api_dump : bool, optional
        If ``True`` (default), enable Vulkan API dump.

    Notes
    -----
    ``validation`` requires validation layer [1]_.
    ``api_dump`` requires LunarG API dump layer [2]_.
    If required layers are not installed, the options are ignored.

    References
    ----------
    .. [1] VK_LAYER_KHRONOS_validation
       https://github.com/KhronosGroup/Vulkan-ValidationLayers
    .. [2] VK_LAYER_LUNARG_api_dump
       https://github.com/LunarG/VulkanTools/blob/main/layersvt/api_dump_layer.md
    """
    wblog.start_logging("vulkpy", level=logging.DEBUG)
    logger.debug("Enable debug mode")

    layers = []
    if validation:
        layers.append("VK_LAYER_KHRONOS_validation")
        logger.debug("Enable Vulkan Validation")
    if api_dump:
        layers.append("VK_LAYER_LUNARG_api_dump")
        logger.debug("Enable Vulkan API dump")

    if len(layers) > 0:
        os.environ["VK_INSTANCE_LAYERS"] = ":".join(layers)


def getShader(name: str):
    """
    Get Shader Path

    Parameters
    ----------
    name : str
        SPIR-V (.spv) name
    """
    return os.path.join(os.path.dirname(__file__), "shader", name)
