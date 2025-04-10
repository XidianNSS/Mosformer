file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE INC_FILES "include/*.h")

add_library(crypto_ass STATIC ${SRC_FILES} ${INC_FILES})
target_include_directories(crypto_ass PUBLIC include ${CMAKE_SOURCE_DIR}/crypto/parameters/include)
target_link_libraries(crypto_ass model_lib common_lib)