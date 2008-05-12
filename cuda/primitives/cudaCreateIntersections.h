/*! \file cudaCreateIntersections.h
 *  \author Jared Hoberock *  \brief Defines the interface to a CUDA kernel
 *         which transforms the results of
 *         cudaRayTriangleBVHIntersection into CudaIntersections.
 */

#pragma once

#include "../../primitives/PrimitiveHandle.h"
#include "CudaIntersection.h"
#include "../geometry/CudaDifferentialGeometryArray.h"

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

extern "C"
  void createIntersectionData(const float4 *rayOriginsAndMinT,
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
                              CudaDifferentialGeometryArray &dg,
                              PrimitiveHandle *hitPrims,
                              const size_t n);

void createIntersectionData(const float3 *origins,
                            const float3 *directions,
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
                            CudaDifferentialGeometryArray &dg,
                            PrimitiveHandle *hitPrims,
                            const size_t n);

