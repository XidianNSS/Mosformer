file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE INC_FILES "include/*.h")

add_library(network STATIC ${SRC_FILES} ${INC_FILES})
target_include_directories(network PUBLIC include ${CMAKE_SOURCE_DIR}/common/include)