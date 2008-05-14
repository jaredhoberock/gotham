/*! \file checkFramebufferStatus.inl
 *  \author Jared Hoberock
 *  \brief Inline file for checkFramebufferStatus.h.
 */

#include "checkFramebufferStatus.h"
#include <iostream>

void checkFramebufferStatus(const char *filename, const unsigned int lineNumber)
{
  GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
  if(status != GL_FRAMEBUFFER_COMPLETE_EXT)
  {
    std::cerr << filename << "(" << lineNumber << "): ";
    switch(status)
    {
      case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
      {
        std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT" << std::endl;
        break;
      } // end case

      case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
      {
        std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT" << std::endl;
        break;
      } // end case

      case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
      {
        std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT" << std::endl;
        break;
      } // end case

      case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
      {
        std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT" << std::endl;
        break;
      } // end case

      case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
      {
        std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT" << std::endl;
        break;
      } // end case

      case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
      {
        std::cerr << "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT" << std::endl;
        break;
      } // end case

      case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
      {
        std::cerr << "GL_FRAMEBUFFER_UNSUPPORTED_EXT" << std::endl;
        break;
      } // end case
    } // end switch
  } // end if
} // end checkFramebufferStatus()


