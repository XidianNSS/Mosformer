file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE INC_FILES "include/*.h")

add_library(model_lib STATIC ${SRC_FILES} ${INC_FILES})
target_include_directories(model_lib PUBLIC include ${CMAKE_SOURCE_DIR}/external)
target_link_libraries(model_lib
    PRIVATE
        crypto_lib
        common_lib
        cnpy
)