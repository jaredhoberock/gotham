/*! \file Buffer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Buffer.h.
 */

#include "Buffer.h"

Buffer::Buffer(void):Parent()
{
  setTarget(GL_ARRAY_BUFFER_ARB);
} // end Buffer::Buffer()

Buffer::Buffer(const GLenum target):Parent(target)
{
  ;
} // end Buffer::Buffer()

void Buffer::init(const GLsizeiptrARB size,
                  const GLenum usage,
                  const void *data)
{
  // bind first
  bind();
  glBufferData(getTarget(), size, data, usage);

  // unbind
  unbind();

  mSize = size;
} // end Buffer::init()

void Buffer::create(const GLenum target)
{
  setTarget(target);
  Parent::create();
} // end Buffer::create()

GLsizeiptrARB Buffer::getSize(void) const
{
  return mSize;
} // end Buffer::getSize()

