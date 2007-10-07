/*! \file MutatorApi.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an api for the
 *         mutators library.
 */

#ifndef MUTATOR_API_H
#define MUTATOR_API_H

#include "../api/Gotham.h"
class PathMutator;

class MutatorApi
{
  public:
    /*! This static method creates a new PathMutator
     *  given the attributes supplied in the given
     *  AttributeMap.
     *  \param attr The attributes describing the PathMutator
     *              to create.
     *  \return A new PathMutator.
     */
    static PathMutator *mutator(const Gotham::AttributeMap &attr);
}; // end MutatorApi

#endif // MUTATOR_API_H

