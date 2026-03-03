cmake_minimum_required(VERSION 3.20)
project(saltro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(SALTRO_BUILD_PYTHON "Build python bindings" ON)
option(SALTRO_BUILD_MANUAL_DEBUG "Build manual debug executables" ON)
option(SALTRO_BUILD_TESTS "Build unit tests" ON)

include(FetchContent)

set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/eigen
)
FetchContent_Populate(eigen)
add_library(Eigen3::Eigen INTERFACE IMPORTED)
target_include_directories(Eigen3::Eigen INTERFACE
    ${CMAKE_SOURCE_DIR}/external/eigen
)

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

    file(GLOB_RECURSE PYTHON_BINDING_SOURCES_TEMP CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/python/*_py.cpp
    )

    # Parse dependencies from PYBIND_DEPENDS comments
    set(BINDINGS_WITH_DEPS "")
    set(BINDINGS_WITHOUT_DEPS "")
    set(DEPENDENCY_MAP "")
    
    foreach(binding ${PYTHON_BINDING_SOURCES_TEMP})
        get_filename_component(fname ${binding} NAME_WE)
        string(REPLACE "_py" "" base ${fname})
        
        file(READ ${binding} binding_content)
        string(REGEX MATCH "// PYBIND_DEPENDS: ([^\n]+)" dep_match "${binding_content}")
        
        if(dep_match)
            string(REGEX REPLACE "// PYBIND_DEPENDS: ([^\n]+)" "\\1" deps "${dep_match}")
            string(STRIP "${deps}" deps)
            # Convert space-separated string to CMake list
            string(REPLACE " " ";" deps_list "${deps}")
            list(APPEND BINDINGS_WITH_DEPS ${binding})
            set(DEPS_${base} ${deps_list})
        else()
            list(APPEND BINDINGS_WITHOUT_DEPS ${binding})
        endif()
    endforeach()
    
    # Build ordered list: dependencies first, then dependents, then rest
    set(PYTHON_BINDING_SOURCES "")
    set(PROCESSED "")
    
    # Add files without dependencies first
    foreach(binding ${BINDINGS_WITHOUT_DEPS})
        list(APPEND PYTHON_BINDING_SOURCES ${binding})
        get_filename_component(fname ${binding} NAME_WE)
        string(REPLACE "_py" "" base ${fname})
        list(APPEND PROCESSED ${base})
    endforeach()
    
    # Add files with dependencies (iterative topological sort)
    set(REMAINING ${BINDINGS_WITH_DEPS})
    set(MAX_ITERATIONS 100)
    set(iteration 0)
    
    while(REMAINING AND iteration LESS MAX_ITERATIONS)
        set(ADDED_THIS_ITERATION FALSE)
        set(NEW_REMAINING "")
        
        foreach(binding ${REMAINING})
            get_filename_component(fname ${binding} NAME_WE)
            string(REPLACE "_py" "" base ${fname})
            
            # Check if all dependencies are satisfied
            set(deps_satisfied TRUE)
            foreach(dep ${DEPS_${base}})
                list(FIND PROCESSED ${dep} dep_idx)
                if(dep_idx EQUAL -1)
                    set(deps_satisfied FALSE)
                    break()
                endif()
            endforeach()
            
            if(deps_satisfied)
                list(APPEND PYTHON_BINDING_SOURCES ${binding})
                list(APPEND PROCESSED ${base})
                set(ADDED_THIS_ITERATION TRUE)
            else()
                list(APPEND NEW_REMAINING ${binding})
            endif()
        endforeach()
        
        set(REMAINING ${NEW_REMAINING})
        
        if(NOT ADDED_THIS_ITERATION AND REMAINING)
            # Circular dependency or missing dependency
            message(FATAL_ERROR "Cannot resolve dependencies for Python bindings. Remaining: ${REMAINING}")
        endif()
        
        math(EXPR iteration "${iteration} + 1")
    endwhile()
    
    if(iteration EQUAL MAX_ITERATIONS)
        message(FATAL_ERROR "Dependency resolution exceeded maximum iterations")
    endif()

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

if(SALTRO_BUILD_TESTS)

    include(CTest)
    enable_testing()

    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.0
    )
    FetchContent_MakeAvailable(catch2)

    file(GLOB_RECURSE UNIT_TEST_SOURCES
        CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/tests/unit/test_*.cpp
    )

    foreach(test_src ${UNIT_TEST_SOURCES})

        # Get path relative to tests/unit
        file(RELATIVE_PATH rel_path
            ${CMAKE_SOURCE_DIR}/tests/unit
            ${test_src}
        )

        # Remove extension
        get_filename_component(rel_no_ext ${rel_path} NAME_WE)

        # Get subdirectory
        get_filename_component(rel_dir ${rel_path} DIRECTORY)

        # Replace "/" with "_" for safe target names
        string(REPLACE "/" "_" target_name ${rel_path})
        string(REPLACE ".cpp" "" target_name ${target_name})

        add_executable(${target_name} ${test_src})

        target_link_libraries(${target_name}
            PRIVATE
            saltro
            Catch2::Catch2WithMain
        )

        target_include_directories(${target_name}
            PRIVATE
            ${CMAKE_SOURCE_DIR}/include
        )

        add_test(NAME ${target_name} COMMAND ${target_name})

        # Optional: preserve folder structure in build directory
        set_target_properties(${target_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
            ${CMAKE_BINARY_DIR}/unit_tests/${rel_dir}
        )

    endforeach()

endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(saltro PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow
        -Wconversion -Wdouble-promotion
    )
endif()