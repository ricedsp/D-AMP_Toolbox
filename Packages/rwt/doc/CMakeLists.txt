# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc
${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)

find_package(LATEX)
find_program(DOXYFILE_MAKE make)
mark_as_advanced(DOXYFILE_MAKE)
add_custom_command(TARGET doc
        POST_BUILD
        COMMAND "${DOXYFILE_MAKE}"
        COMMENT "Running LaTeX for Doxygen documentation in ${CMAKE_CURRENT_SOURCE_DIR}/latex..."
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/latex")


endif(DOXYGEN_FOUND)
