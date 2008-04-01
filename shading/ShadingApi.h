/*! \file ShadingApi.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an
 *         api for instantiating objects
 *         related to shading.
 */

#pragma once

#include "MaterialList.h"
#include "ShadingContext.h"
#include <boost/shared_ptr.hpp>
#include "../api/Gotham.h"

class ShadingApi
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
    static ShadingContext *context(Gotham::AttributeMap &attr,
                                   const boost::shared_ptr<MaterialList> &materials);
}; // end ShadingApi

