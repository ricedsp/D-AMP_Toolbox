cmake_minimum_required (VERSION 2.6)
project (rwt)
subdirs(lib/src)
subdirs(doc)
subdirs(python)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
#FIND_PACKAGE(MatlabMex REQUIRED)

# This section based on http://www.cmake.org/pipermail/cmake/2003-June/003953.html
IF (UNIX)
  ADD_CUSTOM_TARGET (distclean @echo cleaning for source distribution)
  SET(DISTCLEANED
   CMakeFiles
   cmake.depends
   cmake.check_depends
   CMakeCache.txt
   cmake.check_cache
   Makefile
   *.cmake
   */CMakeCache.txt
   */CMakeFiles
   */Makefile
   */*.cmake
   */*/CMakeCache.txt
   */*/CMakeFiles
   */*/*.cmake
   */*/Makefile
   lib/src/*.a
   doc/Doxyfile
   doc/html
   doc/latex
   core core.*
   gmon.out
   */*.mex*
   */*.o
   lib/src/*.o
   python/rwtPYTHON_wrap.cxx
   python/rwt.py
   python/rwt.pyc
   python/_rwt.so
   *~
  )
  
  ADD_CUSTOM_COMMAND(
    DEPENDS clean
    COMMENT "distribution clean"
    COMMAND rm
    ARGS    -Rf CMakeTmp ${DISTCLEANED}
    TARGET  distclean
  )
ENDIF(UNIX)
