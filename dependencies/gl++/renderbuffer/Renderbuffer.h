/*! \file Renderbuffer.h
 *  \author Jared Hoberock
 *  \brief Defines an interface for abstracting OpenGL
 *         Renderbuffer objects.
 */

#ifndef RENDERBUFFER_H
#define RENDERBUFFER_H

#include <GL/glew.h>
#include "../globject/GLObject.h"

/*! \fn getTextureThunk
 *  \todo Find a way around this.
 */
inline void genRenderbufferThunk(GLuint num, GLuint *ids)
{
  glGenRenderbuffersEXT(num,ids);
} // end genRenderbufferThunk()

/*! \fn deleteRenderbufferThunk
 *  \tod Find a way around this.
 */
inline void deleteRenderbufferThunk(GLuint num, GLuint *ids)
{
  glDeleteRenderbuffersEXT(num, ids);
} // end deleteRenderbufferThunk()

/*! \fn bindRenderbufferThunk
 *  \todo Find a way around this.
 */
inline void bindRenderbufferThunk(GLenum target, GLuint id)
{
  glBindRenderbufferEXT(target, id);
} // end bindRenderbufferThunk()

/*! \class Renderbuffer
 *  \brief Encapsulates an OpenGL Renderbuffer object.
 *  \todo Take all the stuff shared with Texture into a
 *        common base.
 */
class Renderbuffer : public GLObject<genRenderbufferThunk,
                                     deleteRenderbufferThunk,
                                     bindRenderbufferThunk>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef GLObject<genRenderbufferThunk,
                     deleteRenderbufferThunk,
                     bindRenderbufferThunk> Parent;

    /*! \fn Renderbuffer
     *  \brief Null constructor calls the parent and sets
     *         mTarget to GL_RENDERBUFFER_EXT.
     *         Also:
     *         - calls setWidth(1)
     *         - calls setHeight(1)
     */
    inline Renderbuffer(void);

    /*! \fn setInternalFormat
     *  \brief This method sets mInternalFormat.
     *  \param f Sets mInternalFormat
     */
    inline void setInternalFormat(const GLenum f);

    /*! \fn getInternalFormat
     *  \brief This method returns mInternalFormat
     *  \return mInternalFormat
     */
    inline GLenum getInternalFormat(void) const;

    /*! \fn getWidth
     *  \brief This method returns mWidth.
     *  \return mWidth
     */
    inline GLsizei getWidth(void) const;

    /*! \fn getHeight
     *  \brief This method returns mHeight.
     *  \return mHeight
     */
    inline GLsizei getHeight(void) const;

    /*! \fn init
     *  \brief This method allocates storage for this Renderbuffer.
     *  \param internalFormat The internal format of this Rendebuffer.
     *  \param width The width in pixels to allocate.
     *  \param height The height in pixels to allocate.
     *  \note Also calls the appropriate methods to set each
     *        member as specified by the parameters.
     */
    inline void init(const GLenum internalFormat,
                     const GLsizei width,
                     const GLsizei height);

  protected:
    /*! \fn setWidth
     *  \brief This method sets mWidth.
     *  \param w Sets mWidth
     */
    inline void setWidth(const GLsizei w);

    /*! \fn setHeight
     *  \brief This method sets mHeight.
     *  \param h Sets mHeight
     */
    inline void setHeight(const GLsizei h);
    
    /*! A Renderbuffer has an internal format.
     */
    GLenum mInternalFormat;

    /*! A Renderbuffer has a width in pixels.
     */
    GLsizei mWidth;

    /*! A Renderbuffer has a height in pixels.
     */
    GLsizei mHeight;
}; // end class Renderbuffer

#include "Renderbuffer.inl"

#endif // RENDERBUFFER_H

