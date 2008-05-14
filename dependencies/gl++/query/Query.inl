/*! \file Query.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Query.h.
 */

#include "Query.h"

Query::Query(void)
  :Parent(GL_SAMPLES_PASSED_ARB)
{
  ;
} // end Query::Query()

bool Query
  ::isAvailable(void) const
{
  GLint available = 0;
  glGetQueryObjectivARB(getIdentifier(),
                        GL_QUERY_RESULT_AVAILABLE_ARB,
                        &available);

  if(available)
  {
    glGetQueryObjectuivARB(getIdentifier(),
                           GL_QUERY_RESULT_ARB,
                           &mResult);
  } // end if

  return available != 0;
} // end Query::isAvailable()

GLuint Query
  ::spinAndGetResult(void) const
{
  while(!isAvailable());

  return getResult();
} // end Query::spinAndGetResult()

GLuint Query
  ::getResult(void) const
{
  return mResult;
} // end Query::getResult()

