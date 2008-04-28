/*! \file cudaCreateIntersections.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a CUDA kernel
 *         which transforms the results of
 *         cudaRayTriangleBVHIntersection into CudaIntersections.
 */

#pragma once

#include "../../primitives/PrimitiveHandle.h"
#include "CudaIntersection.h"

extern "C"
  void cudaCreateIntersections(const float4 *rayOriginsAndMinT,
                               const float4 *rayDirectionsAndMaxT,
                               const float4 *timeBarycentricsAndTriangleIndex,
                               const float3 *geometricNormals,
                               const float3 *firstVertex,
                               const float3 *secondVertex,
                               const float3 *thirdVertex,
                               const float2 *firstVertexParms,
                               const float2 *secondVertexParms,
                               const float2 *thirdVertexParms,
                               const float  *invSurfaceArea,
                               const PrimitiveHandle *primHandles,
                               const bool *stencil,
                               CudaIntersection *intersections,
                               const size_t n);


