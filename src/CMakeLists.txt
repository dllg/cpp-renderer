include(../cmake/deps.cmake)
include(../cmake/opts.cmake)

# ---- Add source files ----
# Note: globbing sources is considered bad practice as CMake's generators may
# not detect new files automatically. Keep that in mind when changing files, or
# explicitly mention them here.
file(GLOB headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.h*")
file(GLOB sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# ---- Create library ----

add_library(${PROJECT_NAME} ${headers} ${sources})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

target_compile_options(${PROJECT_NAME} PUBLIC ${compile_opts})

target_compile_definitions(${PROJECT_NAME} PUBLIC ${defines})

target_link_options(${PROJECT_NAME} PUBLIC ${link_opts})

target_link_libraries(${PROJECT_NAME} PUBLIC ${dependencies})

target_link_directories(${PROJECT_NAME} PUBLIC ${library_dirs})

target_include_directories(${PROJECT_NAME} PUBLIC ${include_dirs}
"${CMAKE_CURRENT_SOURCE_DIR}"
)

