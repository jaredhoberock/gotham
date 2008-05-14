/*! \file Framebuffer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Framebuffer.h.
 */

#include "Framebuffer.h"

Framebuffer::Framebuffer(void):Parent()
{
  setTarget(GL_FRAMEBUFFER_EXT);
} // end Framebuffer::Framebuffer()

void Framebuffer::attachTexture(const GLenum textureTarget,
                                const GLenum attachment,
                                const GLuint texture,
                                const GLint level)
{
  glFramebufferTexture2DEXT(getTarget(), attachment,
                            textureTarget, texture, level);
} // end Framebuffer::attachTexture()

void Framebuffer::attachTextureLayer(const GLenum attachment,
                                     const GLuint texture,
                                     const GLint layer,
                                     const GLint level)
{
#ifdef WIN32
  glFramebufferTextureLayerEXT(getTarget(), attachment, texture, level, layer);
#else
  std::cerr << "Not implemented on Ubuntu yet." << std::endl;
#endif // WIN32
} // end Framebuffer::attachTextureLayer()

void Framebuffer::attachRenderbuffer(const GLenum renderbufferTarget,
                                     const GLenum attachment,
                                     const GLuint renderbuffer)
{
  glFramebufferRenderbufferEXT(getTarget(), attachment,
                               renderbufferTarget,
                               renderbuffer);
} // end Framebuffer::attachRenderbuffer()

void Framebuffer::detach(const GLenum attachment)
{
  glFramebufferTexture2DEXT(getTarget(), attachment,
                            GL_TEXTURE_RECTANGLE_NV,
                            0, 0);
} // end Framebuffer::detach()

bool Framebuffer::isComplete(void) const
{
  return GL_FRAMEBUFFER_COMPLETE_EXT == glCheckFramebufferStatusEXT(getTarget());
} // end Framebuffer::isComplete()

