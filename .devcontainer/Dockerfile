FROM intel/oneapi:latest

# Install dependencies for metrics-discovery
RUN apt-get update && apt-get install -y ninja-build

# setup metrics-discovery for VTune
RUN apt-get update && apt-get -y install libdrm-dev

WORKDIR /metrics-discovery

RUN git clone https://github.com/intel/metrics-discovery.git .

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && cmake --install build
