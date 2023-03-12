FROM python:bullseye AS vulkpy-env
RUN --mount=type=cache,target=/var/lib/apt/lists \
    wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | \
    apt-key add - && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-focal.list \
    http://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list && \
    apt update && \
    apt install -y --no-install-recommends \
    libvulkan1 libvulkan-dev vulkan-headers shaderc \
    vulkan-validationlayers lunarg-vulkan-layers mesa-vulkan-drivers
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install numpy pybind11 well-behaved-logging


FROM vulkpy-env AS vulkpy-install
WORKDIR /vulkpy-ci
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install coverage unittest-xml-reporting
COPY setup.py pyproject.toml MANIFEST.in mypy.ini .
COPY vulkpy vulkpy
RUN --mount=type=cache,target=/root/.cache/pip pip install .[test] && \
    mypy -p vulkpy && \
    rm -rf vulkpy && \
    rm setup.py pyproject.toml MANIFEST.in mypy.ini


FROM vulkpy-install AS vulkpy-test
COPY test test
WORKDIR /vulkpy-ci/test
COPY .coveragerc .
RUN coverage run --source vulkpy -m xmlrunner discover || true
RUN mkdir -p /coverage && cp -v .coverage.* /coverage && \
    mkdir -p /unittest && cp *.xml /unittest


FROM vulkpy-install AS vulkpy-combine
WORKDIR /coverage
RUN --mount=type=cache,target=/root/.cache/pip pip install coverage
COPY vulkpy /vulkpy-ci/vulkpy
COPY .coveragerc .coveragerc
COPY --from=vulkpy-test /coverage /coverage
RUN coverage combine && \
    echo "## Test Coverage\n\`\`\`\n" >> summary.md && \
    coverage report | tee -a summary.md && \
    echo "\n\`\`\`" >> summary.md && \
    mkdir -p /coverage/html && coverage html -d /coverage/html


FROM vulkpy-install AS vulkpy-example
WORKDIR /vulkpy-ci/example
RUN --mount=type=cache,target=/root/.cache/pip pip install scikit-learn
COPY example .
RUN python 00-arithmetic.py && \
    python 01-random.py && \
    python 02-nn.py --debug --optimizer sgd --nepoch 1 && \
    python 02-nn.py --debug --optimizer adam  --nepoch 1 && \
    touch /vulkpy-ci/example/example-ok


FROM vulkpy-env AS vulkpy-build
WORKDIR /build
RUN --mount=type=cache,target=/root/.cache/pip pip install wheel
COPY LICENSE setup.py README.md MANIFEST.in pyproject.toml .
COPY vulkpy vulkpy
RUN python setup.py sdist -d /dist


FROM vulkpy-env AS vulkpy-doc
WORKDIR /ci
RUN --mount=type=cache,target=/var/lib/apt/lists \
    apt update && apt -y --no-install-recommends install graphviz
RUN --mount=type=cache,target=/root/.cache/pip pip install \
    sphinx \
    furo \
    sphinx-automodapi \
    myst-parser
COPY LICENSE LICENSE
COPY setup.py setup.py
COPY README.md README.md
COPY vulkpy vulkpy
RUN --mount=type=cache,target=/root/.cache/pip pip install .[doc]
COPY doc doc
COPY example example
RUN sphinx-build -W -b html doc /html


FROM scratch AS results
COPY --from=vulkpy-test /unittest /unittest/3.11
COPY --from=vulkpy-combine /coverage/html /coverage/html
COPY --from=vulkpy-combine /coverage/summary.md /coverage/summary.md
COPY --from=vulkpy-build /dist /dist
COPY --from=vulkpy-doc /html /html
COPY --from=vulkpy-example /vulkpy-ci/example/example-ok /example/example-ok
CMD [""]
