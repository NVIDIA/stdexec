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
        target_link_libraries(${example_name} PRIVATE CUDASchedulers)

        target_include_directories(${example_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    endforeach()

    add_executable(maxwell_mc maxwell.cpp)
    target_link_libraries(maxwell_mc PRIVATE stdexec)
    target_include_directories(maxwell_mc PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    target_compile_options(maxwell_mc PRIVATE -stdpar=multicore)
    target_link_options(maxwell_mc PRIVATE -stdpar=multicore)
    target_compile_features(maxwell_mc PRIVATE cxx_std_20)
endif()
