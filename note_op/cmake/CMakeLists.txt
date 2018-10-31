# Minimum CMake required
cmake_minimum_required(VERSION 3.5)

if(WIN32)
	if(${CMAKE_VERSION} VERSION_LESS "3.8")
		message(WARNING "Your current cmake version is ${CMAKE_VERSION} which does not support setting the toolset architecture to x64. This may cause \"compiler out of heap space\" errors when building. Consider upgrading your cmake to > 3.8 and using the flag -Thost=x64 when running cmake.")
	else()
		if(NOT CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE OR NOT "${CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE}" STREQUAL "x64")
			message(WARNING "Your current cmake generator is set to use 32 bit toolset architecture. This may cause \"compiler out of heap space\" errors when building. Consider using the flag -Thost=x64 when running cmake.")
		endif()
	endif()
endif()

# Project
project(note_op_proj C CXX)

# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD 11)

# Actual source is the ../src
get_filename_component(note_op_src_dir ${note_op_proj_SOURCE_DIR} PATH)
set(note_op_src_dir "${note_op_src_dir}/src")

# [CLEANUP] Not sure if this is needed (copied from Protobuf)
# CMake policies
cmake_policy(SET CMP0022 NEW)

set(tensorflow_CUDA_VERSION "9.0" CACHE STRING "CUDA version to build against")
set(tensorflow_CUDNN_VERSION "7" CACHE STRING "cuDNN version to build against")

if (NOT WIN32)
  # Default install paths for cuda libraries in Linux
  # In some Linux distros, find_package(CUDA) seems to require CMAKE_LIBRARY_PATH to include cuda-lib paths
  list(APPEND CMAKE_LIBRARY_PATH "${tensorflow_CUDA_LIBRARY_PATH}")
  list(APPEND CMAKE_LIBRARY_PATH "${tensorflow_CUDA_LIBRARY_PATH}/stubs")
endif (NOT WIN32)

find_package(CUDA ${tensorflow_CUDA_VERSION} REQUIRED EXACT)

# by default we assume compute cabability 3.5 and 5.2. If you change this change it in
# CUDA_NVCC_FLAGS and cuda_config.h below
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_37,code=[sm_37,compute_37])
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=[sm_52,compute_52])
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=[sm_60,compute_60])
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=[sm_61,compute_61])
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_70,code=[sm_70,compute_70])
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ftz=true)  # Flush denormals to zero
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-rdc=true)

set(common_config_done "not_done" CACHE STRING "common_config_done")
if (NOT (${common_config_done} STREQUAL "done"))
  set(py_work_dir ${note_op_proj_SOURCE_DIR})
  message("check tensorflow version...")
  execute_process(COMMAND python get_tensorflow_info.py checkDevice
    WORKING_DIRECTORY ${py_work_dir} 
    RESULT_VARIABLE tf_device_result 
    OUTPUT_VARIABLE tf_device)

  message("tensorflow version ${tf_device}")
  if (NOT (${tf_device} STREQUAL "GPU"))
    message(FATAL_ERROR "note op lib need tensorflow-gpu version" )
  endif()

  message("get tensorflow dir and flags...")
  execute_process(COMMAND python get_tensorflow_info.py getIncludeDir
    WORKING_DIRECTORY ${py_work_dir} 
    RESULT_VARIABLE tf_include_dir_result 
    OUTPUT_VARIABLE tf_include_dir_value)

  execute_process(COMMAND python get_tensorflow_info.py getLibDir
    WORKING_DIRECTORY ${py_work_dir} 
    RESULT_VARIABLE tf_lib_dir_result 
    OUTPUT_VARIABLE tf_lib_dir_value)

  execute_process(COMMAND python get_tensorflow_info.py getCompileFlags
    WORKING_DIRECTORY ${py_work_dir} 
    RESULT_VARIABLE tf_cflag_result 
    OUTPUT_VARIABLE tf_cflag_value)

  execute_process(COMMAND python get_tensorflow_info.py getLinkFlags
    WORKING_DIRECTORY ${py_work_dir} 
    RESULT_VARIABLE tf_lflag_result 
    OUTPUT_VARIABLE tf_lflag_value)

  set(tf_include_dir "${tf_include_dir_value}" CACHE STRING "tf_include_dir")
  set(tf_lib_dir "${tf_lib_dir_value}" CACHE STRING "tf_lib_dir")
  set(tf_cflag "${tf_cflag_value}" CACHE STRING "tf_cflag")
  set(tf_lflag "${tf_lflag_value}" CACHE STRING "tf_lflag")
else()
  message("common config have done, skip check tensorflow")
endif()
set(common_config_done "done" CACHE STRING "common_config_done" FORCE)

message("tf_include_dir: ${tf_include_dir}")
message("tf_lib_dir: ${tf_lib_dir}")
message("tf_cflag: ${tf_cflag}")
message("tf_lflag: ${tf_lflag}")

set(custom_op_link_lib_dir "${tf_lib_dir}/python")
set(custom_op_link_lib "pywrap_tensorflow_internal")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${tf_cflag}")

link_directories(${custom_op_link_lib_dir})
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")

if (WIN32)
  # windows下为了解决tensorshape里的std limts max min 和vs的max min的冲突
  add_definitions(-DNOMINMAX)
endif()

set(note_op_cuda_srcs
    "${note_op_src_dir}/hmm_viterbi_op_kernel.cu"
)
set_source_files_properties(${note_op_cuda_srcs} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
set(CUDA_SEPARABLE_COMPILATION ON)
CUDA_ADD_LIBRARY(note_op_cuda ${note_op_cuda_srcs} STATIC)

set(note_op_srcs
    "${note_op_src_dir}/note_op_util.h"
    "${note_op_src_dir}/note_op_util.cc"
    "${note_op_src_dir}/hmm_viterbi_op.h"
    "${note_op_src_dir}/hmm_viterbi_op.cc"
    "${note_op_src_dir}/note_op.cc"
)

add_library(note_op SHARED ${note_op_srcs})

target_link_libraries(note_op 
  "note_op_cuda"
  ${custom_op_link_lib} 
)

get_filename_component(install_dir ${note_op_src_dir} PATH)
set(install_dir "${install_dir}/lib")
install(TARGETS note_op DESTINATION ${install_dir})