if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" OR STDEXEC_ENABLE_CUDA)
    file(GLOB_RECURSE EXAMPLES
            RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
            CONFIGURE_DEPENDS
            *.cpp)

    foreach(example ${EXAMPLES})
        file(TO_CMAKE_PATH "${example}" example_prefix)
        string(REPLACE "/" "." example_prefix "${example_prefix}")
        get_filename_component(example_name "${example_prefix}" NAME_WLE)

        add_executable(${example_name} ${example})

        if("${example}" STREQUAL "maxwell_cpu_mt.cpp")
          target_link_libraries(${example_name} PRIVATE stdexec)
          target_compile_options(${example_name} PRIVATE -stdpar=multicore)
          target_link_options(${example_name} PRIVATE -stdpar=multicore)
          target_compile_features(${example_name} PRIVATE cxx_std_20)
        else()
          target_link_libraries(${example_name} PRIVATE CUDASchedulers)
          target_include_directories(${example_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
        endif()
    endforeach()
endif()

