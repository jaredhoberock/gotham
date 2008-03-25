/*! \file PrimitiveApi.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an api
 *         for instantiating Primitives.
 */

#pragma once

#include "Primitive.h"
#include "PrimitiveList.h"
#include "../api/Gotham.h"

class PrimitiveApi
{
  public:
    /*! This static method creates a new PrimitiveList given
     *  the options in the given AttributeMap.
     *  \param attr An AttributeMap describing a set of
     *         rendering attributes.
     *  \param prims A PrimitiveList to copy from.
     *  \return A new PrimitiveList.
     */
    static PrimitiveList<> *list(Gotham::AttributeMap &attr,
                                 const PrimitiveList<> &prims);
}; // end PrimitiveApi

