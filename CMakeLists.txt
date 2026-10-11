cmake_minimum_required(VERSION 3.18)
project(BronKerbosch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Компилируем только под GPU, установленный в системе.
set(CMAKE_CUDA_ARCHITECTURES native)

# Параллельная сборка внутри одного .cu.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 0")

add_library(bk_seq STATIC bk_sequential.cpp)
target_include_directories(bk_seq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(bk_seq PRIVATE -O3 -Wall -Wextra -Wpedantic)

add_library(bk_gpu STATIC bk_cuda.cu)
target_include_directories(bk_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(bk_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

add_library(gen STATIC graph_generator.cpp)
target_include_directories(gen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(gen PRIVATE -O3 -Wall -Wextra -Wpedantic)

add_executable(bk_test main.cpp)
target_compile_options(bk_test PRIVATE -O3 -Wall -Wextra -Wpedantic)
target_link_libraries(bk_test PRIVATE bk_seq bk_gpu gen)
