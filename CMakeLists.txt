cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

if (NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "Please set VCPKG_ROOT environment variable first.")
endif()

MESSAGE(STATUS "VCPKG_ROOT=$ENV{VCPKG_ROOT}")

SET(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vcpkg toolchain file")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

project(cpp-renderer LANGUAGES CXX C)

add_subdirectory(src)
add_subdirectory(console)
