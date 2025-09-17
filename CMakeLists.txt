cmake_minimum_required(VERSION 3.10)
project(my_project CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/arm64-osx/share/opencv4")

# 明确指定目标架构（针对 Apple Silicon Mac）
if(APPLE AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Target architectures for macOS")
    message(STATUS "Setting default macOS architecture to: ${CMAKE_OSX_ARCHITECTURES}")
endif()

# 使用 vcpkg 工具链
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(DEFINED ENV{VCPKG_ROOT})
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "Vcpkg toolchain file")
        message(STATUS "Using vcpkg toolchain from VCPKG_ROOT: ${CMAKE_TOOLCHAIN_FILE}")
    endif()
endif()

if(CMAKE_TOOLCHAIN_FILE AND EXISTS ${CMAKE_TOOLCHAIN_FILE})
    include(${CMAKE_TOOLCHAIN_FILE})
endif()


# 查找并链接 fmt 库
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(mini_map_splicing main.cpp)
target_link_libraries(mini_map_splicing PRIVATE fmt::fmt ${OpenCV_LIBS})
