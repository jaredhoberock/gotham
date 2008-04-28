/*! \file cudaRayTriangleBVHIntersection.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a function
 *         which performs SIMD ray-triangle intersection
 *         using a BVH in CUDA.
 */

#pragma once

extern "C"
  void cudaRayTriangleBVHIntersection(const unsigned int NULL_NODE,
                                      const unsigned int rootIndex,
                                      const float4 *rayOriginsAndMinT,
                                      const float4 *rayDirectionsAndMaxT,
                                      const float4 *minBoundHitIndex,
                                      const float4 *maxBoundMissIndex,
                                      const float4 *firstVertexDominantAxis,
                                      bool *stencil,
                                      float4 *timeBarycentricsAndTriangleIndex,
                                      const size_t n);

extern "C"
  void cudaShadowRayTriangleBVHIntersectionWithStencil(const unsigned int NULL_NODE,
                                                       const unsigned int rootIndex,
                                                       const float4 *rayOriginsAndMinT,
                                                       const float4 *rayDirectionsAndMaxT,
                                                       const float4 *minBoundHitIndex,
                                                       const float4 *maxBoundMissIndex,
                                                       const float4 *firstVertexDominantAxis,
                                                       const bool *stencil,
                                                       bool *results,
                                                       const size_t n);

