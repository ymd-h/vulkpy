FROM python:bullseye AS vulkpy-build
RUN --mount=type=cache,target=/var/lib/apt/lists \
    wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | \
    apt-key add - && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-focal.list \
    http://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list && \
    apt update && \
    apt install -y \
    libvulkan1 libvulkan-dev vulkan-headers shaderc \
    vulkan-validationlayers lunarg-vulkan-layers
RUN --mount=type=cache,target=/root/.cache/pip pip install numpy well-behaved-logging
