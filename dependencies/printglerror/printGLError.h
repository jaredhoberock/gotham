/*! \file printGLError.h
 *  \author Jared Hoberock
 *  \brief Prints OpenGL error messages to std::err.
 */

#ifndef PRINT_GL_ERROR_H
#define PRINT_GL_ERROR_H

/*! \fn printGLError
 *  \brief Queries OpenGL for the last error message, and prints the name
 *         to std::err if one exists.
 *  \param fileName The source filename generating the error message.
 *  \param lineNumber The number of the line generating the error message.
 *  \note This function is inlined so you don't have to compile against an additional
 *        source file.
 */
inline void printGLError(const char *fileName, const unsigned int lineNumber);

#include "printGLError.inl"

#endif // PRINT_GL_ERROR_H

