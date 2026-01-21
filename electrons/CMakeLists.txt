cmake_minimum_required(VERSION 3.20)
project(electrons LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
add_definitions(-DUNICODE=0)

set(SRC_DIR  ${CMAKE_SOURCE_DIR})
set(INC_DIR  ${CMAKE_SOURCE_DIR})
set(CUDA_DIR ${CMAKE_SOURCE_DIR})
set(DATA_DIR ${CMAKE_SOURCE_DIR})

set(SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/Simulation.cpp
    ${SRC_DIR}/Renderer.cpp

    ${CUDA_DIR}/CudaField.cu
    ${CUDA_DIR}/CudaParticles.cu
)

set(CUDA_PATH $ENV{CUDA_PATH})

add_executable(electrons ${SOURCES})

target_include_directories(electrons PRIVATE
    ${INC_DIR}
    ${SRC_DIR}
    ${CUDA_DIR}
    "C:/SFML-2.6.1/include"
    "${CUDA_PATH}/include"
)

target_link_directories(electrons PRIVATE
    "C:/SFML-2.6.1/lib"
    "${CUDA_PATH}/lib/x64"
)

target_link_libraries(electrons PRIVATE
    cudart_static
    $<$<CONFIG:Debug>:sfml-graphics-d>
    $<$<CONFIG:Debug>:sfml-window-d>
    $<$<CONFIG:Debug>:sfml-system-d>
    $<$<CONFIG:Release>:sfml-graphics>
    $<$<CONFIG:Release>:sfml-window>
    $<$<CONFIG:Release>:sfml-system>
)

set_target_properties(electrons PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)

target_compile_options(electrons PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler=/Zi>
)

add_custom_command(TARGET electrons POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${DATA_DIR} $<TARGET_FILE_DIR:electrons>/data
)
