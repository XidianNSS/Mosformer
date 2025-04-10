file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE INC_FILES "include/*.h")

add_library(application_nn STATIC ${SRC_FILES} ${INC_FILES})
target_include_directories(application_nn PUBLIC include)
target_link_libraries(application_nn
    PRIVATE
        common_lib
        crypto_lib
        model_lib
)