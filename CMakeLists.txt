cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(Mosformer)

find_package(OpenMP REQUIRED)

add_compile_options(-O3)

add_subdirectory(application)
add_subdirectory(common)
add_subdirectory(crypto)
add_subdirectory(secure_model)
add_subdirectory(external/cnpy)

file(GLOB TEST_SOURCES "tests/*.cpp")

include_directories(${CMAKE_SOURCE_DIR}/application/ ${CMAKE_SOURCE_DIR}/common/ ${CMAKE_SOURCE_DIR}/crypto/ ${CMAKE_SOURCE_DIR}/secure_model/ ${CMAKE_SOURCE_DIR}/external)

add_executable(mosformer main.cpp ${TEST_SOURCES})

target_link_libraries(mosformer app_lib common_lib crypto_lib model_lib OpenMP::OpenMP_CXX)
