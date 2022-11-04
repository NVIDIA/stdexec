if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" OR STDEXEC_ENABLE_CUDA)
    file(GLOB_RECURSE EXAMPLES
            RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
            CONFIGURE_DEPENDS
            *.cpp)

    set(compiler_name "")
    get_filename_component(compiler_name "${CMAKE_CXX_COMPILER}" NAME)

    foreach(example ${EXAMPLES})
        file(TO_CMAKE_PATH "${example}" example_prefix)
        string(REPLACE "/" "." example_prefix "${example_prefix}")
        get_filename_component(example_name "${example_prefix}" NAME_WLE)

        if("${example}" STREQUAL "maxwell_cpu_mt.cpp")
          add_executable(${example_name} ${example})
          target_link_libraries(${example_name} PRIVATE stdexec)
          target_compile_options(${example_name} PRIVATE -stdpar=multicore)
          target_link_options(${example_name} PRIVATE -stdpar=multicore)
          target_compile_features(${example_name} PRIVATE cxx_std_20)
        elseif("${example}" STREQUAL "maxwell_distributed.cpp")
          if("${compiler_name}" STREQUAL "mpic++")
            add_executable(maxwell_distributed_ov ${example})
            target_link_libraries(maxwell_distributed_ov PRIVATE CUDASchedulers)
            target_include_directories(maxwell_distributed_ov PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
            target_compile_definitions(maxwell_distributed_ov PRIVATE OVERLAP)

            add_executable(maxwell_distributed ${example})
            target_link_libraries(maxwell_distributed PRIVATE CUDASchedulers)
            target_include_directories(maxwell_distributed PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
          endif()
        else()
          add_executable(${example_name} ${example})
          target_link_libraries(${example_name} PRIVATE CUDASchedulers)
          target_include_directories(${example_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
        endif()
    endforeach()
endif()

