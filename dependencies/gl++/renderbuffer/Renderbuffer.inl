/*! \file Renderbuffer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Renderbuffer.h.
 */

#include "Renderbuffer.h"
#include <printglerror/printGLError.h>

Renderbuffer::Renderbuffer(void):Parent()
{
  setTarget(GL_RENDERBUFFER_EXT);
  setWidth(1);
  setHeight(1);
} // end Renderbuffer::Renderbuffer()

void Renderbuffer::setInternalFormat(const GLenum f)
{
  mInternalFormat = f;
} // end Renderbuffer::setInternalFormat()

GLenum Renderbuffer::getInternalFormat(void) const
{
  return mInternalFormat;
} // end Renderbuffer::getInternalFormat()

GLsizei Renderbuffer::getWidth(void) const
{
  return mWidth;
} // end Renderbuffer::getWidth()

GLsizei Renderbuffer::getHeight(void) const
{
  return mHeight;
} // end Renderbuffer::getHeight()

void Renderbuffer::init(const GLenum internalFormat,
                        const GLsizei width,
                        const GLsizei height)
{
  // bind
  bind();

  // set parameters
  setInternalFormat(internalFormat);
  setWidth(width);
  setHeight(height);

  // malloc
  glRenderbufferStorageEXT(getTarget(), getInternalFormat(),
                           getWidth(), getHeight());

  // unbind
  unbind();
} // end Renderbuffer::init()

void Renderbuffer::setHeight(const GLsizei h)
{
  mHeight = h;
} // end Renderbuffer::setHeight()

void Renderbuffer::setWidth(const GLsizei w)
{
  mWidth = w;
} // end Renderbuffer::setWidth()

