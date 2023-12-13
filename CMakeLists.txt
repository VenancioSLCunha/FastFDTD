cmake_minimum_required (VERSION 3.8)
project(FastFDTD LANGUAGES CXX CUDA)

include_directories(${PROJECT_SOURCE_DIR})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda")

set(Python3_FIND_VIRTUALENV "STANDARD")
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

include(preventBuildInSource)
include(print_target_properties)
include(protect_nvcc_flags)
include(protect_pthread_flag)

#add_library(fdtd_main STATIC src/main.cpp)
add_library(fdtd_cuda STATIC src/solver.cu src/solver_func.h src/solver_matriz.h src/fdtd_timestep.cu)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 12)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

#set_target_properties(fdtd_main PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(fdtd_cuda PROPERTIES CUDA_SEPERABLE_COMPILATION ON POSITION_INDEPENDENT_CODE TRUE CUDA_ARCHITECTURES "75") 

add_subdirectory("pybind11/")
pybind11_add_module(fast_fdtd src/bind_interface.cpp)
#target_link_libraries(fast_fdtd PRIVATE fdtd_main fdtd_cuda)
target_link_libraries(fast_fdtd PRIVATE fdtd_cuda)

message("  ${PROJECT_NAME}")
message("//===============================================//")
message("  CUDA compiler ID      : ${CMAKE_CUDA_COMPILER_ID}")
message("  CUDA compiler Version : ${CMAKE_CUDA_COMPILER_VERSION}")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
  "${CMAKE_CXX_COMPILER_VERSION} "
  "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
message("  CUDA Compile flags : ${CMAKE_CUDA_FLAGS}")
message("  CUDA toolkit inc   : ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message("")
message("  Python3_found       : ${Python3_FOUND}")
message("  Python3 version     : ${Python3_VERSION}")
message("  Python3 interpreter : ${Python3_EXECUTABLE}")
message("  Python3 interp. Id  : ${Python3_INTERPRETER_ID}")
message("  Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message("  Python3_LIBRARY_DIRS: ${Python3_LIBRARY_DIRS}")
