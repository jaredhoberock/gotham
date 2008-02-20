/*! \file PathApi.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an api
 *         for the path library.
 */

#ifndef PATH_API_H
#define PATH_API_H

#include "../api/Gotham.h"
#include "../records/PhotonMap.h"
class PathSampler;

class PathApi
{
  public:
    /*! This static method creates a new PathSampler
     *  given the attributes supplied in the given
     *  AttributeMap.
     *  \param attr The attributes describing the PathSampler
     *              to create.
     *  \param photonMaps A set of PhotonMaps.
     *  \return A new PathSampler.
     */
    static PathSampler *sampler(Gotham::AttributeMap &attr,
                                const Gotham::PhotonMaps &photonMaps);

    /*! This method fills an AttributeMap with this library's defaults.
     *  \param attr The set of attributes to add to.
     */
    static void getDefaultAttributes(Gotham::AttributeMap &attr);
}; // end PathApi

#endif // PATH_API_H

