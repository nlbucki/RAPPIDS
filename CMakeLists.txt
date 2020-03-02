cmake_minimum_required(VERSION 3.2)

project(RAPPIDS_project CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost COMPONENTS program_options REQUIRED)

add_subdirectory(src)
add_subdirectory(test)
