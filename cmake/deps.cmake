# Download yocto-gl from github
include(FetchContent)
FetchContent_Declare(
  yocto-gl
  GIT_REPOSITORY https://github.com/xelatihy/yocto-gl
  GIT_TAG        main
)

FetchContent_GetProperties(yocto-gl)
if(NOT yocto-gl_POPULATED)
  FetchContent_Populate(yocto-gl)
  add_subdirectory(${yocto-gl_SOURCE_DIR} ${yocto-gl_BINARY_DIR})
endif()

message(STATUS "yocto-gl_SOURCE_DIR: ${yocto-gl_SOURCE_DIR}")

find_path(YOCTO_GL_INCLUDE_DIRS "yocto/yocto_sceneio.h" PATHS ${yocto-gl_SOURCE_DIR}/libs)

# Add find_package calls here
find_package(argh CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# Add dependencies here
set(dependencies
    argh
    spdlog::spdlog
)

# Add include directories here
set (include_dirs
  ${YOCTO_GL_INCLUDE_DIRS}
)

# Add library directories here
set (library_dirs
  ""
)
