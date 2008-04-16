/*! \file cudaGetMaterialHandles.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to helper
 *         functions for CudaSurfacePrimitiveList.
 */

#pragma once

#include "../../primitives/PrimitiveHandle.h"
#include "../../shading/MaterialHandle.h"

extern "C" void getMaterialHandles(const PrimitiveHandle *prims,
                                   const int *stencil,
                                   const MaterialHandle *primToMaterial,
                                   MaterialHandle *materials,
                                   const size_t n);

extern "C" void getMaterialHandlesWithStride(const PrimitiveHandle *prims,
                                             const size_t primStride,
                                             const int *stencil,
                                             const MaterialHandle *primToMaterial,
                                             MaterialHandle *materials,
                                             const size_t n);

