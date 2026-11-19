cmake_minimum_required(VERSION 3.16)
project(minigpt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(MINIGPT_NATIVE "Optimize for this machine's CPU (-march=native)" ON)

include(CheckCXXCompilerFlag)
if(MSVC)
  add_compile_options(/W4)
  if(MINIGPT_NATIVE)
    add_compile_options(/arch:AVX2)
  endif()
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
  if(MINIGPT_NATIVE)
    # x86 compilers use -march=native; Apple clang on ARM wants -mcpu=native.
    check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
    check_cxx_compiler_flag(-mcpu=native HAS_MCPU_NATIVE)
    if(HAS_MARCH_NATIVE)
      add_compile_options(-march=native)
    elseif(HAS_MCPU_NATIVE)
      add_compile_options(-mcpu=native)
    endif()
  endif()
endif()

find_package(Threads REQUIRED)

add_library(minigpt_core src/tokenizer.cpp src/model.cpp)
target_include_directories(minigpt_core PUBLIC src)
target_link_libraries(minigpt_core PUBLIC Threads::Threads)

add_executable(minigpt src/main.cpp)
target_link_libraries(minigpt PRIVATE minigpt_core)

add_executable(minigpt-bench src/bench.cpp)
target_link_libraries(minigpt-bench PRIVATE minigpt_core)

enable_testing()
add_executable(minigpt-tests tests/tests.cpp)
target_link_libraries(minigpt-tests PRIVATE minigpt_core)
add_test(NAME unit COMMAND minigpt-tests ${CMAKE_SOURCE_DIR}/models/test)
