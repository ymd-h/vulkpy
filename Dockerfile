FROM python:bullseye
RUN wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | \
    apt-key add - && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-focal.list \
    http://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list && \
    apt update && \
    apt install -y \
    libvulkan1 libvulkan-dev vulkan-headers shaderc vulkan-validationlayers && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

