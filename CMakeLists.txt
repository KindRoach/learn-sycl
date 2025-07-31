cmake_minimum_required(VERSION 3.22)

project(LearnSYCL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(IntelSYCL REQUIRED)

# Get a list of all .cpp files in the root directory
set(src_root "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE sources "${src_root}" "${src_root}/*.cpp")

# Create an executable target for each .cpp file
# as the dir structure in src
foreach(file_path ${sources})

    # Extract the target name and directory from the file path
    get_filename_component(target_name "${file_path}" NAME_WE)
    get_filename_component(target_dir "${file_path}" DIRECTORY)
    file(RELATIVE_PATH target_dir "${src_root}" "${target_dir}")

    add_executable("${target_name}" "${file_path}")
    target_include_directories("${target_name}" PRIVATE src)
    add_sycl_to_target(TARGET "${target_name}" SOURCES "${file_path}")
    set_target_properties("${target_name}" PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${target_dir}"
    )
endforeach()
