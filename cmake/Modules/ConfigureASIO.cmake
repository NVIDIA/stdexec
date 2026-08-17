option(STDEXEC_ENABLE_ASIO "Enable ASIO targets" OFF)
set(STDEXEC_ASIO_IMPLEMENTATION
    "boost"
    CACHE STRING "boost")
set_property(CACHE STDEXEC_ASIO_IMPLEMENTATION PROPERTY STRINGS boost
                                                        standalone)

if(STDEXEC_ENABLE_ASIO)
  set(STDEXEC_ASIO_USES_BOOST FALSE)
  set(STDEXEC_ASIO_USES_STANDALONE FALSE)

  if(${STDEXEC_ASIO_IMPLEMENTATION} STREQUAL "boost")
    set(STDEXEC_ASIO_USES_BOOST TRUE)
  elseif(${STDEXEC_ASIO_IMPLEMENTATION} STREQUAL "standalone")
    set(STDEXEC_ASIO_USES_STANDALONE TRUE)
  else()
    message(FATAL_ERROR "Unknown configuration for ASIO implementation: "
                        ${STDEXEC_ASIO_IMPLEMENTATION})
  endif()

  set(STDEXEC_ASIO_CONFIG_HPP
      ${CMAKE_CURRENT_BINARY_DIR}/include/exec/asio/asio_config.hpp)

  configure_file(include/exec/asio/asio_config.hpp.in
                 ${STDEXEC_ASIO_CONFIG_HPP})

  file(GLOB_RECURSE asioexec_sources CONFIGURE_DEPENDS include/exec/asio/*.hpp)

  if(${STDEXEC_ASIO_USES_BOOST})
    set(BOOST_INCLUDE_LIBRARIES asio system)
    set(BOOST_VERSION 1.86.0)
    rapids_cpm_find(
      Boost
      ${BOOST_VERSION}
      CPM_ARGS
      URL
      https://github.com/boostorg/boost/releases/download/boost-${BOOST_VERSION}/boost-${BOOST_VERSION}-cmake.tar.xz
      OPTIONS
      "BOOST_SKIP_INSTALL_RULES OFF")

    add_library(asioexec INTERFACE)
    list(APPEND stdexec_export_targets asioexec)
    add_library(STDEXEC::asioexec ALIAS asioexec)

    # These aliases are provided for backwards compatibility with the old target
    # names
    add_library(asioexec_boost ALIAS asioexec)
    add_library(stdexec_boost_pool ALIAS asioexec)
    add_library(STDEXEC::asio_pool ALIAS asioexec)
    add_library(STDEXEC::asioexec_boost ALIAS asioexec)

    target_sources(
      asioexec
      PUBLIC FILE_SET
             headers
             TYPE
             HEADERS
             BASE_DIRS
             ${CMAKE_CURRENT_SOURCE_DIR}/include
             FILES
             ${asioexec_sources}
             BASE_DIRS
             ${CMAKE_CURRENT_BINARY_DIR}/include
             FILES
             ${STDEXEC_ASIO_CONFIG_HPP})

    target_compile_definitions(asioexec INTERFACE STDEXEC_ASIO_USES_BOOST)

    target_link_libraries(asioexec INTERFACE STDEXEC::stdexec Boost::asio)
    if(STDEXEC_INSTALL)
      install(
        TARGETS asioexec
        EXPORT stdexec-exports
        FILE_SET headers)
    endif()

  elseif(${STDEXEC_ASIO_USES_STANDALONE})
    include(cmake/import_standalone_asio.cmake)
    import_standalone_asio(TAG "asio-1-31-0" VERSION "1.31.0")

    add_library(asioexec INTERFACE)
    list(APPEND stdexec_export_targets asioexec)
    add_library(STDEXEC::asioexec ALIAS asioexec)

    # These aliases are provided for backwards compatibility with the old target
    # names
    add_library(asioexec_asio ALIAS asioexec)
    add_library(stdexec_asio_pool ALIAS asioexec)
    add_library(STDEXEC::asio_pool ALIAS asioexec)
    add_library(STDEXEC::asioexec_asio ALIAS asioexec)

    target_sources(
      asioexec
      PUBLIC FILE_SET
             headers
             TYPE
             HEADERS
             BASE_DIRS
             ${CMAKE_CURRENT_SOURCE_DIR}/include
             FILES
             ${asioexec_sources}
             BASE_DIRS
             ${CMAKE_CURRENT_BINARY_DIR}/include
             FILES
             ${STDEXEC_ASIO_CONFIG_HPP})

    target_compile_definitions(asioexec INTERFACE STDEXEC_ASIO_USES_STANDALONE)

    target_link_libraries(asioexec INTERFACE STDEXEC::stdexec asio)
    if(STDEXEC_INSTALL)
      install(
        TARGETS asioexec asio
        EXPORT stdexec-exports
        FILE_SET headers)
    endif()

  else()
    message(FATAL_ERROR "ASIO implementation is not configured")
  endif()

  if(STDEXEC_BUILD_TESTS)
    # Verify that every header in exec/asio/ is self-contained, the same way
    # the root CMakeLists.txt does for stdexec's own FILE_SET. Doing it here
    # instead of there matters because these headers need Boost.Asio's (or
    # standalone Asio's) include directories to compile standalone, and
    # asioexec - unlike stdexec - actually links them in (see
    # target_link_libraries(asioexec INTERFACE STDEXEC::stdexec Boost::asio)
    # / ... asio above). This is a real, automatically-checked per-header
    # test, not a hand-written stand-in: CMake generates one standalone
    # compilation per header in asioexec's FILE_SET and folds them into the
    # same top-level all_verify_interface_header_sets target the root
    # CMakeLists.txt's check uses, so both run from a single
    # `cmake --build build --target all_verify_interface_header_sets`. It
    # also means this scales automatically as exec/asio/ grows: no
    # per-file wiring needed here or in the root CMakeLists.txt's
    # exclusion list (see the corresponding SKIP_LINTING there, which is
    # glob-based for the same reason).
    set_target_properties(asioexec PROPERTIES VERIFY_INTERFACE_HEADER_SETS
                                               TRUE)
  endif()
endif()
