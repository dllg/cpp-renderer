# Add compile and link options here

# set(SANITIZER 1)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  if (SANITIZER)
    set(compile_opts -g -fsanitize=address -fsanitize=leak -fno-omit-frame-pointer -Wall -Wextra -Werror)
    set(link_opts -g -fsanitize=address -fsanitize=leak)
  else()
    set(compile_opts -Wall -Wextra)
  endif()
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  if (SANITIZER)
    set(compile_opts ${compile_opts} -static-libasan)
    set(link_opts ${link_opts} -static-libasan)
  endif()
endif()

if (NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(compile_opts ${compile_opts}
      -pedantic
    )
endif()

# Add defines here

set(defines
  NOMINMAX=1
  ASSERT=assert
)
