/*! \file Query.h
 *  \author Jared Hoberock
 *  \brief Defines an interface for abstracting
 *         OpenGL query objects.
 */

#ifndef QUERY_H
#define QUERY_H

#include <GL/glew.h>
#include "../globject/GLObject.h"

/*! \fn genQueryThunk
 *  \todo Find a way around this.
 */
inline void genQueryThunk(GLuint num, GLuint *id)
{
  glGenQueriesARB(num, id);
} // end genQueryThunk()

/*! \fn deleteQueryThunk
 *  \todo Find a way around this.
 */
inline void deleteQueryThunk(GLuint num, GLuint *id)
{
  glDeleteQueriesARB(num, id);
} // end deleteQueryThunk()

/*! \fn bindQueryThunk()
 *  \todo XXX Find a way around this.
 */
inline void bindQueryThunk(GLenum target, GLuint id)
{
  if(id == 0)
  {
    // "unbind"
    glEndQueryARB(target);
  } // end if
  else
  {
    // "bind"
    glBeginQueryARB(target, id);
  } // end else
} // end bindQueryThunk()

class Query
  : public GLObject<genQueryThunk,
                    deleteQueryThunk,
                    bindQueryThunk>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef GLObject<genQueryThunk,
                     deleteQueryThunk,
                     bindQueryThunk> Parent;

    /*! \fn Query
     *  \brief Null constructor sets the target
     *         to GL_SAMPLES_PASSED_ARB and
     *         calls the Parent.
     */
    inline Query(void);

    /*! \fn isAvailable
     *  \brief This method returns whether or not
     *         this Query's result is available yet.
     *  \return As above.
     *  \note If this method returns true, the result
     *        is available through a call to getResult().
     */
    inline bool isAvailable(void) const;

    /*! \fn getResult
     *  \brief This method returns the result of this
     *         Query.
     *  \return mResult.
     *  \note The value returned by this method is
     *        undefined if isAvailable() has not yet
     *        been called and returned true.
     */
    inline GLuint getResult(void) const;

    /*! \fn spinAndGetResult
     *  \brief This method spins until the result of this
     *         Query is available and returns the result.
     *  \return mResult.
     */
    inline GLuint spinAndGetResult(void) const;

  protected:
    /*! The result of this Query.
     */
    mutable GLuint mResult;
}; // end Query

#include "Query.inl"

#endif // QUERY_H

