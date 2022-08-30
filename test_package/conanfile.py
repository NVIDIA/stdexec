import os

from conans import ConanFile, CMake, tools

class P2300TestConan(ConanFile):
    settings = "compiler"
    generators = "cmake"

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def test(self):
        if not tools.cross_building(self):
            os.chdir("bin")
            self.run(".{}test_p2300".format(os.sep))
