option(STDEXEC_ENABLE_ASIO "Enable ASIO targets" OFF)
set(STDEXEC_ASIO_IMPLEMENTATION "boost" CACHE STRING "boost")
set_property(CACHE STDEXEC_ASIO_IMPLEMENTATION PROPERTY STRINGS boost standalone)

if(STDEXEC_ENABLE_ASIO)
  set(STDEXEC_ASIO_USES_BOOST FALSE)
  set(STDEXEC_ASIO_USES_STANDALONE FALSE)

  if(${STDEXEC_ASIO_IMPLEMENTATION} STREQUAL "boost")
    set(STDEXEC_ASIO_USES_BOOST TRUE)
  elseif(${STDEXEC_ASIO_IMPLEMENTATION} STREQUAL "standalone")
    set(STDEXEC_ASIO_USES_STANDALONE TRUE)
  else()
    message(FATAL_ERROR "Unknown configuration for ASIO implementation: " ${STDEXEC_ASIO_IMPLEMENTATION})
  endif()

  set(ASIOEXEC_USES_BOOST ${STDEXEC_ASIO_USES_BOOST})
  set(ASIOEXEC_USES_STANDALONE ${STDEXEC_ASIO_USES_STANDALONE})

  set(STDEXEC_ASIO_POOL_CONFIG_HPP ${CMAKE_CURRENT_BINARY_DIR}/include/execpools/asio/asio_config.hpp)
  set(ASIOEXEC_CONFIG_HPP ${CMAKE_CURRENT_BINARY_DIR}/include/asioexec/asio_config.hpp)

  configure_file(
    include/execpools/asio/asio_config.hpp.in
    ${STDEXEC_ASIO_POOL_CONFIG_HPP}
  )
  configure_file(
    include/asioexec/asio_config.hpp.in
    ${ASIOEXEC_CONFIG_HPP}
  )

  file(GLOB_RECURSE boost_pool_sources CONFIGURE_DEPENDS include/execpools/asio/*.hpp)
  file(GLOB_RECURSE asioexec_sources CONFIGURE_DEPENDS include/asioexec/*.hpp)

  if(${STDEXEC_ASIO_USES_BOOST})
    set(BOOST_INCLUDE_LIBRARIES asio system)
    set(BOOST_VERSION 1.86.0)
    rapids_cpm_find(Boost ${BOOST_VERSION}
      CPM_ARGS
        URL https://github.com/boostorg/boost/releases/download/boost-${BOOST_VERSION}/boost-${BOOST_VERSION}-cmake.tar.xz
        OPTIONS "BOOST_SKIP_INSTALL_RULES OFF"
    )

    add_library(stdexec_boost_pool INTERFACE)
    list(APPEND stdexec_export_targets stdexec_boost_pool)
    add_library(STDEXEC::asio_pool ALIAS stdexec_boost_pool)

    target_sources(stdexec_boost_pool PUBLIC
      FILE_SET headers
      TYPE HEADERS
      BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
      FILES ${boost_pool_sources}
      BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/include
      FILES ${STDEXEC_ASIO_POOL_CONFIG_HPP}
    )

    target_compile_definitions(stdexec_boost_pool INTERFACE STDEXEC_ASIO_USES_BOOST)

    target_link_libraries(stdexec_boost_pool INTERFACE
      STDEXEC::stdexec
      Boost::asio
    )
    install(TARGETS stdexec_boost_pool
      EXPORT stdexec-exports
      FILE_SET headers
    )

    add_library(asioexec_boost INTERFACE)
    list(APPEND stdexec_export_targets asioexec_boost)
    add_library(STDEXEC::asioexec_boost ALIAS asioexec_boost)

    target_sources(asioexec_boost PUBLIC
      FILE_SET headers
      TYPE HEADERS
      BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
      FILES ${asioexec_sources}
      BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/include
      FILES ${ASIOEXEC_CONFIG_HPP}
    )

    target_compile_definitions(asioexec_boost INTERFACE STDEXEC_ASIO_USES_BOOST)

    target_link_libraries(asioexec_boost INTERFACE
      STDEXEC::stdexec
      Boost::asio
    )
    install(TARGETS asioexec_boost EXPORT stdexec-exports FILE_SET headers)

  elseif(${STDEXEC_ASIO_USES_STANDALONE})
    include(cmake/import_standalone_asio.cmake)
    import_standalone_asio(
      TAG "asio-1-31-0"
      VERSION "1.31.0")

    add_library(stdexec_asio_pool INTERFACE)
    list(APPEND stdexec_export_targets stdexec_asio_pool)
    add_library(STDEXEC::asio_pool ALIAS stdexec_asio_pool)

    target_sources(stdexec_asio_pool PUBLIC
      FILE_SET headers
      TYPE HEADERS
      BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
      FILES ${boost_pool_sources}
      BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/include
      FILES ${STDEXEC_ASIO_POOL_CONFIG_HPP}
    )

    target_compile_definitions(stdexec_asio_pool INTERFACE STDEXEC_ASIO_USES_STANDALONE)

    target_link_libraries(stdexec_asio_pool INTERFACE
      STDEXEC::stdexec
      asio
    )
    install(TARGETS stdexec_asio_pool EXPORT stdexec-exports FILE_SET headers)

    add_library(asioexec_asio INTERFACE)
    list(APPEND stdexec_export_targets asioexec_asio)
    add_library(STDEXEC::asioexec_asio ALIAS asioexec_asio)

    target_sources(asioexec_asio PUBLIC
      FILE_SET headers
      TYPE HEADERS
      BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
      FILES ${asioexec_sources}
      BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/include
      FILES ${ASIOEXEC_CONFIG_HPP}
    )

    target_compile_definitions(asioexec_asio INTERFACE STDEXEC_ASIO_USES_STANDALONE)

    target_link_libraries(asioexec_asio INTERFACE
      STDEXEC::stdexec
      asio
    )
    install(TARGETS asioexec_asio EXPORT stdexec-exports FILE_SET headers)
  else()
    message(FATAL_ERROR "ASIO implementation is not configured")
  endif()
endif()
