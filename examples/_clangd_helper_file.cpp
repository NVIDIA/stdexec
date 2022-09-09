// A file that sorts to the top of clangd's file list to help clangd infer the compile commands for nearby headers.
//
// To provide intellisense when a user opens a header file, clangd searches the current dir, subdirs, and parent dirs
// for nearby translation units. It sorts the list of nearby TUs, and uses the compile string of the first TU to
// provide intellisense for the opened header file.
//
// Clangd's heuristic is not perfect, but we can help it along. By starting this filename with an underscore, we
// ensure it sorts to the top of the list of TUs.
//
// TUs closer to the header file in the directory tree are chosen before ones further up or down the directory tree,
// therefore this file should be copied into any subdirectories with tranlation units whose compile flags are different
// than those in the parent (for example, if the subdirectory has a CMakeList.txt that defines additional executables).
// This ensures clangd provides useful intellisense for headers in any subdirectory with a CMakeList.txt.

int main(void) {}
