# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/catherinejohnson/Desktop/Sites/CSCI_2270/matrices_ml

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/catherinejohnson/Desktop/Sites/CSCI_2270/matrices_ml/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/matrices_ml.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/matrices_ml.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/matrices_ml.dir/flags.make

CMakeFiles/matrices_ml.dir/main.cpp.o: CMakeFiles/matrices_ml.dir/flags.make
CMakeFiles/matrices_ml.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/catherinejohnson/Desktop/Sites/CSCI_2270/matrices_ml/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/matrices_ml.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/matrices_ml.dir/main.cpp.o -c /Users/catherinejohnson/Desktop/Sites/CSCI_2270/matrices_ml/main.cpp

CMakeFiles/matrices_ml.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matrices_ml.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/catherinejohnson/Desktop/Sites/CSCI_2270/matrices_ml/main.cpp > CMakeFiles/matrices_ml.dir/main.cpp.i

CMakeFiles/matrices_ml.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matrices_ml.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/catherinejohnson/Desktop/Sites/CSCI_2270/matrices_ml/main.cpp -o CMakeFiles/matrices_ml.dir/main.cpp.s

CMakeFiles/matrices_ml.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/matrices_ml.dir/main.cpp.o.requires

CMakeFiles/matrices_ml.dir/main.cpp.o.provides: CMakeFiles/matrices_ml.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/matrices_ml.dir/build.make CMakeFiles/matrices_ml.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/matrices_ml.dir/main.cpp.o.provides

CMakeFiles/matrices_ml.dir/main.cpp.o.provides.build: CMakeFiles/matrices_ml.dir/main.cpp.o


# Object files for target matrices_ml
matrices_ml_OBJECTS = \
"CMakeFiles/matrices_ml.dir/main.cpp.o"

# External object files for target matrices_ml
matrices_ml_EXTERNAL_OBJECTS =

matrices_ml: CMakeFiles/matrices_ml.dir/main.cpp.o
matrices_ml: CMakeFiles/matrices_ml.dir/build.make
matrices_ml: CMakeFiles/matrices_ml.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/catherinejohnson/Desktop/Sites/CSCI_2270/matrices_ml/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable matrices_ml"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matrices_ml.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matrices_ml.dir/build: matrices_ml

.PHONY : CMakeFiles/matrices_ml.dir/build

CMakeFiles/matrices_ml.dir/requires: CMakeFiles/matrices_ml.dir/main.cpp.o.requires

.PHONY : CMakeFiles/matrices_ml.dir/requires

CMakeFiles/matrices_ml.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/matrices_ml.dir/cmake_clean.cmake
.PHONY : CMakeFiles/matrices_ml.dir/clean

CMakeFiles/matrices_ml.dir/depend:
	cd /Users/catherinejohnson/Desktop/Sites/CSCI_2270/matrices_ml/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/catherinejohnson/Desktop/Sites/CSCI_2270/matrices_ml /Users/catherinejohnson/Desktop/Sites/CSCI_2270/matrices_ml /Users/catherinejohnson/Desktop/Sites/CSCI_2270/matrices_ml/cmake-build-debug /Users/catherinejohnson/Desktop/Sites/CSCI_2270/matrices_ml/cmake-build-debug /Users/catherinejohnson/Desktop/Sites/CSCI_2270/matrices_ml/cmake-build-debug/CMakeFiles/matrices_ml.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/matrices_ml.dir/depend

