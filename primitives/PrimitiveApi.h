/*! \file PrimitiveApi.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an api
 *         for instantiating Primitives.  */

#pragma once

#include "Primitive.h"
#include "PrimitiveList.h"
#include "../api/Gotham.h"
class Scene;

class PrimitiveApi
{
  public:
    /*! This static method creates a new PrimitiveList 
     *  according to options in the given AttributeMap.
     *  \param attr An AttributeMap describing a set of
     *         rendering attributes.
     *  \param prims A PrimitiveList to copy from.
     *  \return A new PrimitiveList.
     */
    static PrimitiveList *list(Gotham::AttributeMap &attr,
                               const PrimitiveList &prims);

    /*! This static method creates a new Scene 
     *  according to the options in the given AttributeMap.
     *  \param attr An AttributeMap describing a set of
     *         rendering attributes.
     *  \return A new Scene.
     */
    static Scene *scene(Gotham::AttributeMap &attr);
}; // end PrimitiveApi

