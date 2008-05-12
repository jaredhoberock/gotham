/*! \file CudaShadingApi.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an
 *         API for instantiating objects
 *         related to shading.
 */

#pragma once

#include <gotham/shading/MaterialList.h>
#include "CudaShadingContext.h"
#include "../api/CudaGotham.h"

class CudaShadingApi
{
  public:
    /*! This static method creates a new ShadingContext
     *  given the options in the given AttributeMap and
     *  a list of Materials.
     *  \param attr An AttributeMap describing a set of
     *              rendering attributes.
     *  \param materials A MaterialList to copy from.
     *  \return A new ShadingContext.
     */
    static CudaShadingContext *context(Gotham::AttributeMap &attr,
                                       const boost::shared_ptr<MaterialList> &materials);
}; // end CudaShadingApi

