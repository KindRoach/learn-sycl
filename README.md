# Learn SYCL (Intel oneAPI)

SYCL examples and micro-benchmarks using Intel oneAPI DPC++ (`icpx`).

## 1. What’s inside

- `.devcontainer/` — VS Code Dev Container setup based on `intel/oneapi:latest`
- `.vscode/` — VS Code run/debug helpers (CMake Tools + GDB oneAPI)
- `src/` — examples grouped by topic (`001-basic`, `002-device`, `003-memory`, ...)
- `CMakeLists.txt` — builds **one executable per `.cpp` file** under `src/`

Output layout:

- Each `src/<topic>/<name>.cpp` becomes `bin/<topic>/<name>` inside your chosen build directory.
	Example: `src/002-device/query-xmx-info.cpp` → `build-release/bin/002-device/query-xmx-info`

Dependencies used by the CMake project:

- IntelSYCL: `find_package(IntelSYCL REQUIRED)`
- oneMKL: `find_package(MKL REQUIRED)` (linked for `matrix-multiply`)

## 2. Requirements

### Toolchain

- Intel oneAPI DPC++ compiler (`icpx`) and SYCL runtime
- oneMKL (oneAPI)
- CMake
- Ninja or Make

### For GPU runs (optional)

- Intel GPU drivers/runtime on the host
- Linux: access to `/dev/dri` (the Dev Container passes this through)

### For debugging in VS Code (optional)

- VS Code extensions: **CMake Tools** and **C/C++**
- oneAPI debugger package (provides `gdb-oneapi`), or adjust the debugger path to your system GDB

## 3. Setup

### Clone Dependency 

This repo expects the dependency folder to exist at `cpp-bench-utils/` (repo root).

```bash
git clone https://github.com/KindRoach/cpp-bench-utils.git
```

### a) VS Code Dev Container (recommended)

This repo includes a Dev Container that starts from `intel/oneapi:latest` and installs a few extras
needed for VTune/metrics.

1) Install VS Code + the “Dev Containers” extension.
2) Open the repo folder in VS Code.
3) Run: “Dev Containers: Reopen in Container”.

Notes:

- Linux devcontainer uses `--device=/dev/dri` and `--privileged`.
- WSL devcontainer uses `--device=/dev/dgx` and mounts `/usr/lib/wsl`.

### b) Local setup

Install Intel oneAPI (Base Toolkit typically includes DPC++ and MKL), then load the environment:

```bash
source /opt/intel/oneapi/setvars.sh
```

Verify:

```bash
icpx --version
cmake --version
```

## 4. How to build, Run and Debug

### Cmake Configure

This repo uses CMake configure presets from `CMakePresets.json`:

```bash
cmake --preset release
```

Other useful presets:

- `cmake --preset debug`
- `cmake --preset relwithdebinfo`

### Build

```bash
cmake --build build-release
```

For debug builds:

```bash
cmake --build build-debug
```

### Run

Executables are written under `build-*/bin/<category>/`.
Some examples:
```bash
./build-release/bin/001-basic/hello-world
./build-release/bin/002-device/query-xmx-info
./build-release/bin/005-matrix/matrix-multiply
```

### Debug with VS Code (recommended)

Install the “CMake Tools” extension, then:

1. Select a configure preset (e.g. `debug` or `relwithdebinfo`).
2. Configure + build.
3. Set the target executable as the debug program (from `build-*/bin/...`).

If you use `relwithdebinfo`, you get a good balance of debug symbols and runtime speed.

### Optional: SYCL AOT (Ahead-Of-Time)

By default AOT is disabled. Enable it by setting `SYCL_DEVICE`:

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-aot -G Ninja \
	-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
	-DSYCL_DEVICE=dg2
cmake --build build-aot
```

Device names are compiler/version specific, please refer to Intel's [doc](
https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-2/ahead-of-time-compilation.html).

### Optional: VTune backend

If you have VTune installed, you can start its local web backend via script:

```bash
./vtune-backend.sh
```

Then open `http://localhost:8080`.
