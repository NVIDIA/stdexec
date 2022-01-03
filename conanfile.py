from conans import ConanFile, CMake, tools
import re
from os import path

class P2300Recipe(ConanFile):
   name = "P2300"
   description = "std::execution"
   author = "Michał Dominiak, Lewis Baker, Lee Howes, Kirk Shoop, Michael Garland, Eric Niebler, Bryce Adelstein Lelbach"
   topics = ("WG21", "concurrency")
   homepage = "https://github.com/brycelelbach/wg21_p2300_std_execution"
   url = "https://github.com/brycelelbach/wg21_p2300_std_execution"
   license = "Apache 2.0"

   settings = "os", "compiler", "build_type", "arch"
   generators = "cmake"
   build_policy = "missing"   # Some of the dependencies don't have builds for all our targets

   options = {"shared": [True, False], "fPIC": [True, False]}
   default_options = {"shared": False, "fPIC": True, "catch2:with_main": True}

   exports_sources = ("include/*", "CMakeLists.txt")

   def set_version(self):
      # Get the version from the spec file
      content = tools.load(path.join(self.recipe_folder, "std_execution.bs"))
      rev = re.search(r"Revision: (\d+)", content).group(1).strip()
      self.version = f"0.{rev}.0"

   def build_requirements(self):
      self.build_requires("catch2/2.13.6")
      # May be needed later
      # self.build_requires("rapidcheck/20210107")
      # self.build_requires("benchmark/1.5.3")

   def config_options(self):
       if self.settings.os == "Windows":
           del self.options.fPIC

   def build(self):
      # Note: options "shared" and "fPIC" are automatically handled in CMake
      cmake = self._configure_cmake()
      cmake.build()

   def package(self):
      cmake = self._configure_cmake()
      cmake.install()

   def package_info(self):
      self.cpp_info.libs = self.collect_libs()

   def _configure_cmake(self):
      cmake = CMake(self)
      if self.settings.compiler == "Visual Studio" and self.options.shared:
         cmake.definitions["CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS"] = True
      cmake.configure(source_folder=None)
      return cmake


