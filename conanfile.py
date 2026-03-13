from conan import ConanFile
from conan.tools.build.cppstd import check_min_cppstd
from conan.tools.cmake import CMake, cmake_layout
from conan.tools.files import copy
from conan.tools.scm import Git

class StdexecPackage(ConanFile):
  name = "p2300"
  description = "std::execution"
  author = "Michał Dominiak, Lewis Baker, Lee Howes, Kirk Shoop, Michael Garland, Eric Niebler, Bryce Adelstein Lelbach"
  topics = ("WG21", "concurrency")
  homepage = "https://github.com/NVIDIA/stdexec"
  url = "https://github.com/NVIDIA/stdexec"
  license = "Apache 2.0"

  settings = "os", "arch", "compiler", "build_type"
  options = {
    "system_context": [True, False],
  }
  default_options = {
    "system_context": False,
  }
  exports_sources = (
    "include/*",
    "src/*",
    "test/*",
    "examples/*",
    "cmake/*",
    "CMakeLists.txt"
  )
  generators = "CMakeToolchain"

  def configure(self):
    if self.options.system_context:
      self.package_type = "static-library"
    else:
      self.package_type = "header-library"

  def validate(self):
    check_min_cppstd(self, "20")

  def set_version(self):
    if not self.version:
      git = Git(self, self.recipe_folder)
      self.version = git.get_commit()

  def layout(self):
    cmake_layout(self)

  def build(self):
    tests = "OFF" if self.conf.get("tools.build:skip_test", default=False) else "ON"
    system_context = "ON" if self.options.system_context else "OFF"

    cmake = CMake(self)
    cmake.configure(variables={
      "STDEXEC_BUILD_TESTS": tests,
      "STDEXEC_BUILD_EXAMPLES": tests,
      "STDEXEC_BUILD_SYSTEM_CONTEXT": system_context,
    })
    cmake.build()
    cmake.test()

  def package_id(self):
    if not self.info.options.system_context:
      # Clear settings because this package is header-only.
      self.info.clear()

  def package(self):
    cmake = CMake(self)
    cmake.install()

  def package_info(self):
    self.cpp_info.set_property("cmake_file_name", "P2300")
    self.cpp_info.set_property("cmake_target_name", "P2300::P2300")
    self.cpp_info.set_property("cmake_target_aliases", ["STDEXEC::stdexec"])
    if self.options.system_context:
      self.cpp_info.components["system_context"].libs = ["system_context"]
      self.cpp_info.components["system_context"].set_property(
        "cmake_target_name", "STDEXEC::system_context"
      )
