cmake_minimum_required(VERSION 3.18)
project(
   face
   LANGUAGES Fortran)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(NETCDF REQUIRED)

message(STATUS "netcdf_f_incdir=${netcdf_f_incdir}")
message(STATUS "netcdf_f_libs=${netcdf_f_libs}")

add_subdirectory(${CMAKE_SOURCE_DIR}/src/)
