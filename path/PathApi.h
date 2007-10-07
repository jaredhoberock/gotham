/*! \file PathApi.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an api
 *         for the path library.
 */

#ifndef PATH_API_H
#define PATH_API_H

#include "../api/Gotham.h"
class PathSampler;

class PathApi
{
  public:
    /*! This static method creates a new PathSampler
     *  given the attributes supplied in the given
     *  AttributeMap.
     *  \param attr The attributes describing the PathSampler
     *              to create.
     *  \return A new PathSampler.
     */
    static PathSampler *sampler(const Gotham::AttributeMap &attr);
}; // end PathApi

#endif // PATH_API_H

