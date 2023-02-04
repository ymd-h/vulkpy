import os


def enable_debug():
    os.environ["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
