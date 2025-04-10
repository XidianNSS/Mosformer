file(GLOB_RECURSE SOURCES "parameter/src/*.cpp" "parameter/include/*.h")

add_subdirectory(arithmetic_secret_sharing)
add_subdirectory(replicated_secret_sharing)

add_library(crypto_lib INTERFACE ${SOURCES})

set(PARAM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/parameters/include)

target_include_directories(crypto_lib INTERFACE ${PARAM_INCLUDE})

target_link_libraries(crypto_lib INTERFACE crypto_ass crypto_rss common_lib app_lib model_lib)
