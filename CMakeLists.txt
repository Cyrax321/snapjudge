cmake_minimum_required(VERSION 3.20)
project(snapjudge VERSION 0.1.0 LANGUAGES CXX)

# snapjudge: C++17 port of the snapjudge System-1 decision engine.
# Self-contained: system BLAS only (Apple Accelerate / OpenBLAS), no ONNX, no libtorch.

option(SNAPJUDGE_CUDA "Build the CUDA fast path (port of snapjudge tl_kernels.py)" OFF)
option(SNAPJUDGE_BUILD_TESTS "Build the doctest suites" ON)
option(SNAPJUDGE_PCRE2 "Use system PCRE2 for the email module (else FetchContent)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# ---- BLAS: Apple Accelerate or OpenBLAS ------------------------------------
if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  set(SNAPJUDGE_BLAS ${ACCELERATE_FRAMEWORK})
  set(SNAPJUDGE_HAS_CBLAS ON)
  # Accelerate's cblas headers live in the framework; newer SDKs also ship
  # <Accelerate/Accelerate.h>. We declare the tiny cblas subset we call ourselves
  # in tensor.cpp, so no include path is needed beyond the framework link.
else()
  find_library(OPENBLAS_LIB NAMES openblas blas REQUIRED)
  set(SNAPJUDGE_BLAS ${OPENBLAS_LIB})
  set(SNAPJUDGE_HAS_CBLAS ON)
endif()

find_package(Threads REQUIRED)
find_package(CURL)          # hub.cpp + client remote mode
find_package(PkgConfig QUIET)
if(SNAPJUDGE_PCRE2 AND PkgConfig_FOUND)
  pkg_check_modules(PCRE2 libpcre2-8)
endif()
if(NOT PCRE2_FOUND)
  find_library(PCRE2_LIBRARIES NAMES pcre2-8)
  find_path(PCRE2_INCLUDE_DIRS pcre2.h)
  if(PCRE2_LIBRARIES AND PCRE2_INCLUDE_DIRS)
    set(PCRE2_FOUND TRUE)
  endif()
endif()

if(SNAPJUDGE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# ---- generated Unicode tables ------------------------------------------------
# The committed file `generated/unicode_tables.inc` is preferred: pure C++ build.
# `python/gen_unicode.py` regenerates it from Python's unicodedata when needed
# (dev-time only), e.g. after upgrading the Unicode version.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/generated/unicode_tables.inc)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)
  set(UNICODE_INC ${CMAKE_CURRENT_BINARY_DIR}/generated/unicode_tables.inc)
  add_custom_command(
    OUTPUT ${UNICODE_INC}
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/generated/unicode_tables.inc ${UNICODE_INC}
    COMMENT "snapjudge: Unicode tables (committed snapshot)"
  )
else()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(UNICODE_INC ${CMAKE_CURRENT_BINARY_DIR}/generated/unicode_tables.inc)
  add_custom_command(
    OUTPUT ${UNICODE_INC}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/gen_unicode.py
            ${UNICODE_INC}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/python/gen_unicode.py
    COMMENT "snapjudge: generating Unicode tables (NFC, properties)"
  )
endif()
add_custom_target(snapjudge_unicode_tables DEPENDS ${UNICODE_INC})

# ---- core library ----------------------------------------------------------
file(GLOB SNAPJUDGE_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(snapjudge STATIC ${SNAPJUDGE_SOURCES})
add_library(snapjudge::snapjudge ALIAS snapjudge)
add_dependencies(snapjudge snapjudge_unicode_tables)

target_include_directories(snapjudge
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    third_party
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_link_libraries(snapjudge PUBLIC ${SNAPJUDGE_BLAS} Threads::Threads)
if(CURL_FOUND)
  target_link_libraries(snapjudge PRIVATE CURL::libcurl)
  target_compile_definitions(snapjudge PRIVATE SNAPJUDGE_HAVE_CURL=1)
endif()
if(PCRE2_FOUND)
  target_link_libraries(snapjudge PRIVATE ${PCRE2_LIBRARIES})
  target_include_directories(snapjudge PRIVATE ${PCRE2_INCLUDE_DIRS})
  target_compile_definitions(snapjudge PRIVATE SNAPJUDGE_HAVE_PCRE2=1)
else()
  message(FATAL_ERROR "snapjudge requires PCRE2 (brew install pcre2 / apt install libpcre2-dev)")
endif()

if(SNAPJUDGE_CUDA)
  add_subdirectory(cuda)
  target_link_libraries(snapjudge PRIVATE snapjudge_cuda CUDA::cudart CUDA::cublas)
  target_compile_definitions(snapjudge PUBLIC SNAPJUDGE_WITH_CUDA=1)
endif()

# ---- binaries (each appears when its module is written) ----------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cli/main.cpp)
  add_executable(snapjudge-cli cli/main.cpp)
  set_target_properties(snapjudge-cli PROPERTIES OUTPUT_NAME snapjudge)
  target_link_libraries(snapjudge-cli PRIVATE snapjudge)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/serve_main/main.cpp)
  add_executable(snapjudge-serve serve_main/main.cpp)
  target_link_libraries(snapjudge-serve PRIVATE snapjudge)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/mcp_main/main.cpp)
  add_executable(snapjudge-mcp mcp_main/main.cpp)
  target_link_libraries(snapjudge-mcp PRIVATE snapjudge)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/bench.cpp)
  add_executable(snapjudge-bench bench/bench.cpp)
  target_link_libraries(snapjudge-bench PRIVATE snapjudge)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/train_main/main.cpp)
  add_executable(snapjudge-train train_main/main.cpp)
  target_link_libraries(snapjudge-train PRIVATE snapjudge)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/train_main/push.cpp)
  add_executable(snapjudge-train-push train_main/push.cpp)
  if(CURL_FOUND)
    target_link_libraries(snapjudge-train-push PRIVATE CURL::libcurl)
  else()
    message(FATAL_ERROR "snapjudge-train-push requires libcurl")
  endif()
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/eval_main/main.cpp)
  add_executable(snapjudge-eval eval_main/main.cpp)
  target_link_libraries(snapjudge-eval PRIVATE snapjudge)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/plots_main/main.cpp)
  add_executable(snapjudge-plots plots_main/main.cpp plots_main/canvas.cpp third_party/lodepng.cpp)
  target_include_directories(snapjudge-plots PRIVATE include third_party)
  target_link_libraries(snapjudge-plots PRIVATE snapjudge)
endif()

# ---- tiny synthetic checkpoint fixture for train tests ----------------------
# Snapshotted at tests/fixtures/tiny-ckpt (built once by python/make_tiny_ckpt.py;
# regenerate that dir by running the python tool when encoder configs change).
set(TINY_CKPT ${CMAKE_CURRENT_BINARY_DIR}/tiny-ckpt)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tiny-ckpt/model.safetensors)
  add_custom_target(snapjudge_tiny_ckpt
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tiny-ckpt ${TINY_CKPT}
    COMMENT "snapjudge: copying committed tiny checkpoint fixture"
  )
else()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_custom_command(
    OUTPUT ${TINY_CKPT}/model.safetensors
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/make_tiny_ckpt.py ${TINY_CKPT}
    DEPENDS python/make_tiny_ckpt.py
    COMMENT "snapjudge: synthesizing tiny checkpoint fixture"
  )
  add_custom_target(snapjudge_tiny_ckpt DEPENDS ${TINY_CKPT}/model.safetensors)
endif()

# ---- regression fixture generator (snapjudge's own outputs locked in) -------
add_executable(gen_fixtures tests/gen_fixtures.cpp)
target_link_libraries(gen_fixtures PRIVATE snapjudge)
target_include_directories(gen_fixtures PRIVATE third_party include)
add_dependencies(gen_fixtures snapjudge_tiny_ckpt)
set_target_properties(gen_fixtures PROPERTIES EXCLUDE_FROM_ALL FALSE)

# ---- tests ------------------------------------------------------------------
if(SNAPJUDGE_BUILD_TESTS)
  enable_testing()
  file(GLOB SNAPJUDGE_TEST_SOURCES CONFIGURE_DEPENDS tests/test_*.cpp)
  foreach(test_src ${SNAPJUDGE_TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE snapjudge)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      ENVIRONMENT "SNAPJUDGE_GOLDEN_DIR=tests/golden")
  endforeach()
  foreach(t test_train test_tokenizer test_tensor test_model)
    if(TARGET ${t})
      add_dependencies(${t} snapjudge_tiny_ckpt)
      target_compile_definitions(${t} PRIVATE SNAPJUDGE_TINY_CKPT="${TINY_CKPT}")
    endif()
  endforeach()
endif()

install(TARGETS snapjudge ARCHIVE DESTINATION lib)
install(DIRECTORY include/snapjudge DESTINATION include)
foreach(b snapjudge-cli snapjudge-serve snapjudge-mcp snapjudge-bench)
  if(TARGET ${b})
    install(TARGETS ${b} RUNTIME DESTINATION bin)
  endif()
endforeach()
