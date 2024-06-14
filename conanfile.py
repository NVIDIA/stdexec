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
  exports_sources = "include/*"
  generators = "CMakeToolchain"

  def validate(self):
    check_min_cppstd(self, "20")

  def set_version(self):
    if not self.version:
      git = Git(self, self.recipe_folder)
      self.version = git.get_commit()

  def layout(self):
    cmake_layout(self)

  def build(self):
    if not self.conf.get("tools.build:skip_test", default=False):
      cmake = CMake(self)
      cmake.configure()
      cmake.build()
      cmake.test()

  def package_id(self):
    # Clear settings because this package is header-only.
    self.info.clear()

  def package(self):
    copy(self, "*.hpp", self.source_folder, self.package_folder)

  def package_info(self):
    self.cpp_info.set_property("cmake_file_name", "P2300")
    self.cpp_info.set_property("cmake_target_name", "P2300::P2300")

    # Clear bin and lib dirs because this package is header-only.
    self.cpp_info.bindirs = []
    self.cpp_info.libdirs = []
