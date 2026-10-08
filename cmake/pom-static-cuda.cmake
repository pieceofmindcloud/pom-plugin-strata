# Links Strata against the CUDA libraries the way the POM links llama.cpp:
# the static cudart, cuBLAS and cuBLASLt of the build image's toolkit. The
# engine then needs only the NVIDIA driver (libcuda.so.1) at run time, which is
# all a POM node image provides - no CUDA shared library is added to it.
#
# Passed with -DCMAKE_PROJECT_INCLUDE: it runs after every project() call, so
# the work is deferred to the end of the top-level CMakeLists, when Strata has
# called find_package(CUDAToolkit) and created CUDA::cudart and CUDA::cublas.
# Strata's sources are not edited.
include_guard(GLOBAL)

function(_pom_redirect target static_target)
  if(NOT TARGET ${target})
    return()
  endif()
  if(NOT TARGET ${static_target})
    message(FATAL_ERROR "pom-static-cuda: ${static_target} is missing from this CUDA toolkit")
  endif()
  get_target_property(_location ${static_target} IMPORTED_LOCATION)
  get_target_property(_interface ${static_target} INTERFACE_LINK_LIBRARIES)
  if(NOT _location)
    message(FATAL_ERROR "pom-static-cuda: ${static_target} has no library file")
  endif()
  if(NOT _interface)
    set(_interface "")
  endif()
  set_target_properties(${target} PROPERTIES
    IMPORTED_LOCATION "${_location}"
    INTERFACE_LINK_LIBRARIES "${_interface}")
  message(STATUS "pom-static-cuda: ${target} -> ${_location}")
endfunction()

function(_pom_static_cuda)
  _pom_redirect(CUDA::cudart CUDA::cudart_static)
  _pom_redirect(CUDA::cublas CUDA::cublas_static)
  _pom_redirect(CUDA::cublasLt CUDA::cublasLt_static)
endfunction()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL _pom_static_cuda)
endif()
