/*! \file Buffer.h
 *  \author Jared Hoberock
 *  \brief Defines an interface for abstracting OpenGL
 *         buffer objects.
 */

#ifndef BUFFER_H
#define BUFFER_H

#include <GL/glew.h>
#include "../globject/GLObject.h"

/*! \fn genBufferThunk
 *  \todo Find a way around this.
 */
inline void genBufferThunk(GLuint num, GLuint *id)
{
  glGenBuffers(num, id);
} // end genBufferThunk()

/*! \fn deleteBufferThunk
 *  \todo Find a way around this.
 */
inline void deleteBufferThunk(GLuint num, GLuint *ids)
{
  glDeleteBuffers(num, ids);
} // end deleteBufferThunk()

/*! \fn bindBufferThunk
 *  \todo Find a way around this.
 */
inline void bindBufferThunk(GLenum target, GLuint id)
{
  glBindBuffer(target, id);
} // end bindBufferThunk()

class Buffer : public GLObject<genBufferThunk,
                               deleteBufferThunk,
                               bindBufferThunk>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef GLObject<genBufferThunk, deleteBufferThunk, bindBufferThunk> Parent;

    /*! \fn Buffer
     *  \brief Null constructor calls the Parent and sets
     *         mTarget to GL_ARRAY_BUFFER_ARB.
     */
    inline Buffer(void);

    /*! \fn Buffer
     *  \param target Sets the target of this Buffer object.
     */
    inline Buffer(const GLenum target);

    /*! \fn create
     *  \brief Creates an OpenGL identifier for this Buffer
     *         and sets its target to the one given.
     *  \param target Sets the target of this Buffer.
     *         GL_ARRAY_BUFFER_ARB by default.
     */
    inline void create(const GLenum target = GL_ARRAY_BUFFER_ARB);

    /*! \fn init
     *  \brief This method allocates memory for this Buffer.
     *  \param size The size of the buffer (in bytes) to allocate.
     *  \param usage A usage hint for this Buffer.
     *  \param data Optional data to copy into the Buffer.
     *              Defaults to 0.
     *  \note This method sets mSize as a side effect.
     */
    inline void init(const GLsizeiptrARB size,
                     const GLenum usage,
                     const void *data = 0);

    /*! \fn getSize
     *  \brief This method returns mSize.
     */
    inline GLsizeiptrARB getSize(void) const;

  protected:
    /*! The size of this Buffer, in bytes.
     */
    GLsizeiptrARB mSize;
}; // end class Buffer

#include "Buffer.inl"

#endif // BUFFER_H

