cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(renderer LANGUAGES CXX C)

include(../cmake/opts.cmake)

# ---- Create binary ----

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(${PROJECT_NAME} ${sources})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

target_compile_options(${PROJECT_NAME} PUBLIC ${compile_opts})

target_compile_definitions(${PROJECT_NAME} PUBLIC ${defines})

target_link_options(${PROJECT_NAME} PUBLIC ${link_opts})

target_link_libraries(${PROJECT_NAME} PUBLIC ${dependencies} cpp-renderer)

target_link_directories(${PROJECT_NAME} PUBLIC ${library_dirs})

target_include_directories(${PROJECT_NAME} PUBLIC ${include_dirs}
"${CMAKE_CURRENT_SOURCE_DIR}"
)
