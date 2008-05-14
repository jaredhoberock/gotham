/*! \file checkFramebufferStatus.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a function to check GL for the status of
 *         the framebuffer.
 */

#ifndef CHECK_FRAMEBUFFER_STATUS_H
#define CHECK_FRAMEBUFFER_STATUS_H

#include <GL/glew.h>

/*! \fn checkFramebufferStatus
 *  \brief This function checks GL for the status of the currently bound
 *         framebuffer and prints a status message if it is not GL_FRAMEBUFFER_COMPLETE_EXT.
 *  \param The name of the file from which this function is called.
 *  \param lineNumber The number of the line from which this function is called
 */
inline void checkFramebufferStatus(const char *filename, const unsigned int lineNumber);

#include "checkFramebufferStatus.inl"

#endif // CHECK_FRAMEBUFFER_STATUS_H

