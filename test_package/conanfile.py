from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout
from conan.tools.build import can_run

class StdexecTestPackage(ConanFile):
  settings = "os", "arch", "compiler", "build_type"
  generators = "CMakeDeps", "CMakeToolchain"

  def requirements(self):
    self.requires(self.tested_reference_str)

  def build(self):
    cmake = CMake(self)
    cmake.configure()
    cmake.build()
    cmake.test()

  def layout(self):
    cmake_layout(self)

  def test(self):
    if can_run(self):
      CMake(self).test()
