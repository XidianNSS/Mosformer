file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE INC_FILES "include/*.h")

add_subdirectory(network)
add_subdirectory(ring)

add_library(common_lib STATIC ${INC_FILES} ${SRC_FILES})
target_include_directories(common_lib PUBLIC include)
target_link_libraries(common_lib network ring)
