file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE INC_FILES "include/*.h")

add_library(ring STATIC ${SRC_FILES} ${INC_FILES})
target_include_directories(ring PUBLIC include ${CMAKE_SOURCE_DIR}/common/include)