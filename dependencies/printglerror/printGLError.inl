/*! \file printGLError.inl
 *  \author Jared Hoberock
 *  \brief Inline file for printGLError.h.
 */

#include <GL/glu.h>
#include <iostream>

/*! \fn printGLError
 *  \brief This function is not null only in _DEBUG mode.
 */
void printGLError(const char *filename, const unsigned int lineNumber)
{
#ifdef _DEBUG
  unsigned int error = glGetError();
  if(error != GL_NO_ERROR)
  {
    std::cerr << "printGLError(): " << filename << "(" << lineNumber << "): ";

    const GLubyte *errorString = gluErrorString(error);

    if(errorString != 0)
    {
      std::cerr << errorString << std::endl;
    } // end if
    else
    {
      std::cerr << "GLU doesn't know wtf." << std::endl;
    } // end else
  } // end if
#endif // _DEBUG
} // end printGLError()

