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
     *  \param photonMaps A set of PhotonMaps.
     *  \return A new PathMutator.
     */
    static PathMutator *mutator(Gotham::AttributeMap &attr,
                                const Gotham::PhotonMaps &photonMaps);

    /*! This method fills an AttributeMap with this library's defaults.
     *  \param attr The set of attributes to add to.
     */
    static void getDefaultAttributes(Gotham::AttributeMap &attr);
}; // end MutatorApi

#endif // MUTATOR_API_H

