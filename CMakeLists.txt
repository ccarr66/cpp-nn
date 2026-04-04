cmake_minimum_required(VERSION 3.15)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

project(cpp-nn VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Bring in cpp-core if a parent project has not already done so.
# When used as a submodule, the parent adds cpp-core first.
# When used standalone, cpp-core lives next to this repo.
if(NOT TARGET cpp-core)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cpp-core
                     ${CMAKE_CURRENT_BINARY_DIR}/cpp-core-build)
endif()

add_library(cpp-nn STATIC
    src/Matrix.cpp
    src/Layer.cpp
    src/NeuralNet.cpp
    src/NNStateCommon.cpp
    src/NNStateRecorder.cpp
)

# PUBLIC: consumers see both cpp-nn and (transitively) cpp-core headers
target_include_directories(cpp-nn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(cpp-nn PUBLIC cpp-core)
