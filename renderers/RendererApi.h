/*! \file RendererApi.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an api
 *         for instantiating Renderers.
 */

#ifndef RENDERER_API_H
#define RENDERER_API_H

#include "Renderer.h"
#include "../api/Gotham.h"

class RendererApi
{
  public:
    /*! This static method creates a new Renderer given
     *  the options in the given AttributeMap.
     *  \param attr An AttributeMap describing a set of
     *              rendering attributes.
     */
    static Renderer *renderer(Gotham::AttributeMap &attr);

    /*! This method fills an AttributeMap with this library's defaults.
     *  \param attr The set of attributes to add to.
     */
    static void getDefaultAttributes(Gotham::AttributeMap &attr);
}; // end RendererApi

#endif // RENDERER_API_H

