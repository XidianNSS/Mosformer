add_subdirectory(nn)

add_library(app_lib INTERFACE)
target_link_libraries(app_lib
    INTERFACE
        application_nn
)