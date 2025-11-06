cmake_minimum_required(VERSION 3.15)
project(F90_CPP_Python_Prototype LANGUAGES C CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python3
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")

# Find pybind11
# First try to find pybind11 using CMake config
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    # If not found, try using Python to locate pybind11
    message(STATUS "pybind11 not found via CMake, trying Python...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_PYTHON_RESULT
    )

    if(PYBIND11_PYTHON_RESULT EQUAL 0)
        message(STATUS "Found pybind11 via Python at: ${PYBIND11_CMAKE_DIR}")
        set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Please install it: pip install pybind11")
    endif()
endif()

message(STATUS "pybind11 version: ${pybind11_VERSION}")
message(STATUS "pybind11 includes: ${pybind11_INCLUDE_DIRS}")

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/src/cpp)

# ============================================
# C++ Library (Python bridge)
# ============================================
add_library(python_bridge STATIC
    src/cpp/python_bridge.cpp
    src/cpp/python_bridge.hpp
)

target_include_directories(python_bridge PUBLIC
    ${CMAKE_SOURCE_DIR}/src/cpp
)

target_link_libraries(python_bridge
    pybind11::embed
)

# ============================================
# Fortran Main Executable
# ============================================
add_executable(fortran_main
    src/fortran/fortran_main.f90
)

target_link_libraries(fortran_main
    python_bridge
    pybind11::embed
)
