cmake_minimum_required(VERSION 3.20)
project(saltro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(SALTRO_BUILD_PYTHON "Build python bindings" ON)
option(SALTRO_BUILD_MANUAL_DEBUG "Build manual debug executables" ON)

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
        ${CMAKE_SOURCE_DIR}/python/*_py.cpp
    )

    if(NOT PYTHON_BINDING_SOURCES)
        message(WARNING "No Python binding sources found in python/")
    else()

        # -----------------------------
        # Generate module.cpp automatically
        # -----------------------------

        set(GENERATED_MODULE_CPP ${CMAKE_BINARY_DIR}/saltro_generated_module.cpp)

        file(WRITE ${GENERATED_MODULE_CPP}
            "#include <pybind11/pybind11.h>\n"
            "namespace py = pybind11;\n\n"
        )

        # Forward declarations
        foreach(binding ${PYTHON_BINDING_SOURCES})

            get_filename_component(fname ${binding} NAME_WE)
            string(REPLACE "_py" "" base ${fname})

            file(APPEND ${GENERATED_MODULE_CPP}
                "void bind_${base}(py::module_&);\n"
            )

        endforeach()

        file(APPEND ${GENERATED_MODULE_CPP}
            "\nPYBIND11_MODULE(saltro_py, m)\n{\n"
            "    m.doc() = \"saltro python bindings\";\n"
        )

        # Call all binders
        foreach(binding ${PYTHON_BINDING_SOURCES})

            get_filename_component(fname ${binding} NAME_WE)
            string(REPLACE "_py" "" base ${fname})

            file(APPEND ${GENERATED_MODULE_CPP}
                "    bind_${base}(m);\n"
            )

        endforeach()

        file(APPEND ${GENERATED_MODULE_CPP}
            "}\n"
        )

        # Build module
        pybind11_add_module(saltro_py
            ${PYTHON_BINDING_SOURCES}
            ${GENERATED_MODULE_CPP}
        )

        target_link_libraries(saltro_py PRIVATE saltro)

    endif()

endif()

if(SALTRO_BUILD_MANUAL_DEBUG)

    file(GLOB_RECURSE DEBUG_TEST_SOURCES
        ${CMAKE_SOURCE_DIR}/tests/debug/debug_*.cpp
    )

    foreach(test_src ${DEBUG_TEST_SOURCES})

        # Get path relative to tests/debug
        file(RELATIVE_PATH rel_path
            ${CMAKE_SOURCE_DIR}/tests/debug
            ${test_src}
        )

        # Remove .cpp extension
        get_filename_component(rel_path_no_ext ${rel_path} NAME_WE)

        # Get directory relative to tests/debug
        get_filename_component(rel_dir ${rel_path} DIRECTORY)

        # Create executable name (replace / with _ to avoid target name issues)
        string(REPLACE "/" "_" target_name ${rel_path_no_ext})

        add_executable(${target_name} ${test_src})

        target_link_libraries(${target_name}
            PRIVATE
            saltro
            Eigen3::Eigen
        )

        target_include_directories(${target_name}
            PRIVATE
            ${CMAKE_SOURCE_DIR}/include
        )

        # Preserve folder structure in build directory
        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
            ${CMAKE_BINARY_DIR}/debug_tests/${rel_dir}
        )

    endforeach()

endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(saltro PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow
        -Wconversion -Wdouble-promotion
    )
endif()