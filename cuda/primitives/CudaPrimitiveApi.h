/*! \file CudaPrimitiveApi.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an
 *         API for creating CudaPrimitives.
 */

#pragma once

#include "CudaPrimitive.h"
#include "../../primitives/PrimitiveList.h"
#include "../api/CudaGotham.h"

class CudaScene;

class CudaPrimitiveApi
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

    /*! This static method creates a new SurfacePrimitiveList
     *  according to options in the given AttributeMap.
     *  \param attr An AttributeMap describing a set of
     *         rendering attributes.
     *  \param surfaces A SurfacePrimitiveList to copy from.
     *  \return A new SurfacePrimitiveList.
     */
    static SurfacePrimitiveList *surfacesList(Gotham::AttributeMap &attr,
                                              const SurfacePrimitiveList &surfaces);

    /*! This static method creates a new CudaScene 
     *  according to the options in the given AttributeMap.
     *  \param attr An AttributeMap describing a set of
     *         rendering attributes.
     *  \return A new CudaScene.
     */
    static CudaScene *scene(Gotham::AttributeMap &attr);
}; // end CudaPrimitiveApi

