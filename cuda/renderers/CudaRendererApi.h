/*! \file CudaRendererApi.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an API
 *         for instantiating CudaRenderers.
 */

#pragma once

#include <gotham/renderers/Renderer.h>
#include "../api/CudaGotham.h"

class CudaRendererApi
{
  public:
    /*! This static method creates a new Renderer given
     *  the options in the given AttributeMap.
     *  \param attr An AttributeMap describing a set of
     *              rendering attributes.
     *  \param photonMaps The set of PhotonMaps available to the Renderer.
     */
    static Renderer *renderer(Gotham::AttributeMap &attr,
                              const Gotham::PhotonMaps &photonMaps);

    /*! This method fills an AttributeMap with this library's defaults.
     *  \param attr The set of attributes to add to.
     */
    static void getDefaultAttributes(Gotham::AttributeMap &attr);
}; // end CudaRendererApi

