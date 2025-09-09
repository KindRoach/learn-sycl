# Require at least CMake 3.22 to ensure support for SYCL features
cmake_minimum_required(VERSION 3.22)

# Define the project name and specify it uses C++
project(LearnSYCL LANGUAGES CXX)

# Use C++20 standard and generate compile_commands.json (useful for tooling)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Intel oneAPI SYCL and MKL packages
find_package(IntelSYCL REQUIRED)
find_package(MKL REQUIRED)

# Define root source directory and gather all .cpp files recursively
set(src_root "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE sources "${src_root}/*.cpp")

# Allow the user to specify a device selector for SYCL compilation
# Please refer to below link for available device name
# https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-2/ahead-of-time-compilation.html
set(SYCL_DEVICE "bmg,mtl" CACHE STRING "SYCL device selector (default: bmg,mtl)")

# Define Ahead-Of-Time (AOT) compilation flags for SYCL targeting Intel GPU
set(SYCL_AOT_FLAGS
        -fsycl-targets=spir64_gen
        -Xsycl-target-backend=spir64_gen
        "-device ${SYCL_DEVICE}"
)

# For each .cpp source file, create an individual executable
# Preserves directory structure relative to src/
foreach (file_path ${sources})

    # Extract executable name (from filename without extension)
    get_filename_component(target_name "${file_path}" NAME_WE)
    # Extract the directory of the file
    get_filename_component(target_dir "${file_path}" DIRECTORY)
    # Compute relative path to src root for output placement
    file(RELATIVE_PATH target_dir "${src_root}" "${target_dir}")

    # Define executable target from the source file
    add_executable("${target_name}" "${file_path}")

    # Include the src directory for headers
    target_include_directories("${target_name}" PRIVATE src)

    # Enable SYCL support for this target
    add_sycl_to_target(TARGET "${target_name}" SOURCES "${file_path}")

    # Apply SYCL AOT compilation and link flags
    target_compile_options("${target_name}" PRIVATE "${SYCL_AOT_FLAGS}")
    target_link_options("${target_name}" PRIVATE "${SYCL_AOT_FLAGS}")

    # Set output directory to bin/<relative path of src>
    set_target_properties("${target_name}" PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${target_dir}"
    )
endforeach ()

# Explicitly link the matrix-multiply target with oneMKL (DPC++ backend)
target_link_libraries(matrix-multiply PRIVATE MKL::MKL_DPCPP)
