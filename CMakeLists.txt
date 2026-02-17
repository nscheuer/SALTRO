cmake_minimum_required(VERSION 3.20)
project(saltro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(SALTRO_BUILD_PYTHON "Build python bindings" ON)
option(SALTRO_BUILD_MANUAL_TESTS "Build manual test executables" ON)

include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/eigen
)
FetchContent_MakeAvailable(eigen)

if(SALTRO_BUILD_PYTHON)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.12.0
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/pybind11
    )
    FetchContent_MakeAvailable(pybind11)
endif()

file(GLOB_RECURSE SALTRO_SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

if(NOT SALTRO_SOURCES)
    message(FATAL_ERROR "No source files found in src/")
endif()

add_library(saltro STATIC ${SALTRO_SOURCES})

target_include_directories(saltro PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(saltro PUBLIC Eigen3::Eigen)

if(SALTRO_BUILD_PYTHON)
    file(GLOB_RECURSE PYTHON_BINDING_SOURCES
        ${CMAKE_SOURCE_DIR}/python/*.cpp
    )

    if(NOT PYTHON_BINDING_SOURCES)
        message(WARNING "No Python binding sources found in python/")
    else()

        pybind11_add_module(saltro_py
            ${PYTHON_BINDING_SOURCES}
        )

        target_link_libraries(saltro_py PRIVATE saltro)
    endif()
endif()

if(SALTRO_BUILD_MANUAL_TESTS)

    file(GLOB_RECURSE MANUAL_TEST_SOURCES
        ${CMAKE_SOURCE_DIR}/tests/manual/*.cpp
    )

    foreach(test_src ${MANUAL_TEST_SOURCES})

        get_filename_component(test_name ${test_src} NAME_WE)

        add_executable(${test_name} ${test_src})

        target_link_libraries(${test_name}
            PRIVATE
            saltro
            Eigen3::Eigen
        )

        target_include_directories(${test_name}
            PRIVATE
            ${CMAKE_SOURCE_DIR}/include
        )

        set_target_properties(${test_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/manual_tests
        )

    endforeach()

endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(saltro PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow
        -Wconversion -Wdouble-promotion
    )
endif()