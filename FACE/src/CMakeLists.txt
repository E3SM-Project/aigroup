set(FACE_SRC main.f90 initial_conditions.f90 config.f90)

include_directories(${netcdf_f_incdir})

add_executable(face ${FACE_SRC})

find_package(FTorch REQUIRED)

target_link_libraries(face PRIVATE ${netcdf_f_libs})
target_link_libraries(face PRIVATE FTorch::ftorch)

install(TARGETS face DESTINATION bin)
