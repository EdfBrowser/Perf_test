# fetch libbutterworth lib
cmake_policy(PUSH)
cmake_policy(SET CMP0135 NEW)

include(FetchContent)

FetchContent_Declare(
  libbutterworth
  GIT_REPOSITORY https://github.com/jorabold/libbutterworth.git
)

FetchContent_MakeAvailable(libbutterworth)

cmake_policy(POP)
