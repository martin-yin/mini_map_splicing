cmake_minimum_required(VERSION 3.10)
project(my_project CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Target architectures")

# 查找并链接 fmt 库
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(mini_map_splicing main.cpp)
target_link_libraries(mini_map_splicing PRIVATE fmt::fmt ${OpenCV_LIBS})
