/*! \file cudaRayTriangleBVHIntersection.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a function
 *         which performs SIMD ray-triangle intersection
 *         using a BVH in CUDA.
 */

#pragma once

void cudaRayTriangleBVHIntersection(const unsigned int NULL_NODE,
                                    const unsigned int rootIndex,
                                    const float3 *origins,
                                    const float3 *directions,
                                    const float2 *intervals,
                                    const float4 *minBoundHitIndex,
                                    const float4 *maxBoundMissIndex,
                                    const float4 *firstVertexDominantAxis,
                                    bool *stencil,
                                    float4 *timeBarycentricsAndTriangleIndex,
                                    const size_t n);

void cudaRayTriangleBVHIntersection(const unsigned int NULL_NODE,
                                    const unsigned int rootIndex,
                                    const float3 *origins,
                                    const float3 *directions,
                                    const float2 &interval,
                                    const float4 *minBoundHitIndex,
                                    const float4 *maxBoundMissIndex,
                                    const float4 *firstVertexDominantAxis,
                                    bool *stencil,
                                    float4 *timeBarycentricsAndTriangleIndex,
                                    const size_t n);


void cudaShadowRayTriangleBVHIntersectionWithStencil(const unsigned int NULL_NODE,
                                                     const unsigned int rootIndex,
                                                     const float3 *rayOrigins,
                                                     const float3 *rayDirections,
                                                     const float2 *rayIntervals,
                                                     const float4 *minBoundHitIndex,
                                                     const float4 *maxBoundMissIndex,
                                                     const float4 *firstVertexDominantAxis,
                                                     const bool *stencil,
                                                     bool *results,
                                                     const size_t n);

void cudaShadowRayTriangleBVHIntersectionWithStencil(const unsigned int NULL_NODE,
                                                     const unsigned int rootIndex,
                                                     const float3 *rayOrigins,
                                                     const float3 *rayDirections,
                                                     const float2 &rayInterval,
                                                     const float4 *minBoundHitIndex,
                                                     const float4 *maxBoundMissIndex,
                                                     const float4 *firstVertexDominantAxis,
                                                     const bool *stencil,
                                                     bool *results,
                                                     const size_t n);

