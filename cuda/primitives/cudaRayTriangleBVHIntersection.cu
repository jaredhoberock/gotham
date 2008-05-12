/*! \file cudaRayTriangleBVHIntersection.cu
 *  \author Jared Hoberock
 *  \brief Implementation of cudaRayTriangleBVHIntersection function.
 */

#include <stdio.h>

#include "cudaRayTriangleBVHIntersection.h"
#include <waldbikkerintersection/cudaWaldBikkerIntersection.h>
#include <stdcuda/vector_math.h>

inline __device__ bool intersectBox(const float3 &o,
                                    const float3 &invDir,
                                    const float3 &minBounds,
                                    const float3 &maxBounds,
                                    const float &tMin,
                                    const float &tMax)
{
  float3 tMin3, tMax3;
  tMin3 = (minBounds - o) * invDir;
  tMax3 = (maxBounds - o) * invDir;

  float3 tNear3 = fminf(tMin3, tMax3);
  float3 tFar3  = fmaxf(tMin3, tMax3);

  float tNear = fmaxf(fmaxf(tNear3.x, tNear3.y), tNear3.z);
  float tFar  = fminf(fminf( tFar3.x,  tFar3.y),  tFar3.z);

  bool hit = tNear <= tFar;
  return hit && tMax >= tNear && tMin <= tFar;
} // end intersectBox()

__global__ void rbk(const unsigned int NULL_NODE,
                    const unsigned int rootIndex,
                    const float3 *origins,
                    const float3 *directions,
                    const float2 interval,
                    const float4 *minBoundHitIndex,
                    const float4 *maxBoundMissIndex,
                    const float4 *firstVertexDominantAxis,
                    bool *stencil,
                    float4 *timeBarycentricsAndTriangleIndex)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(stencil[i])
  {
    float3 origin = origins[i];
    float3 dir = directions[i];
    float2 myInterval = interval;

    float3 invDir = make_float3(1.0f / dir.x,
                                1.0f / dir.y,
                                1.0f / dir.z);
    unsigned int currentNode = rootIndex;
    bool hit = false;
    bool result = false;
    float t = myInterval.y;;
    unsigned int tri = 0;
    float b1 = -1, b2 = -1;
    float4 minBoundsHit, maxBoundsMiss;
    float4 v0Axis;

    // XXX PERF: it might be possible to eliminate these temporaries
    float tempT, tempB1, tempB2;
    while(currentNode != NULL_NODE)
    {
      minBoundsHit = minBoundHitIndex[currentNode];
      maxBoundsMiss = maxBoundMissIndex[currentNode];

      // leaves (primitives) are listed before interior nodes
      // so bounding boxes occur after the root index
      if(currentNode >= rootIndex)
      {
        hit = intersectBox(origin,
                           invDir,
                           make_float3(minBoundsHit.x,
                                       minBoundsHit.y,
                                       minBoundsHit.z),
                           make_float3(maxBoundsMiss.x,
                                       maxBoundsMiss.y,
                                       maxBoundsMiss.z),
                           interval.x,
                           interval.y);
      } // end if
      else
      {
        v0Axis = firstVertexDominantAxis[currentNode];

        hit = cudaWaldBikkerIntersection
          (origin,
           dir,
           myInterval.x, myInterval.y,
           make_float3(v0Axis.x,
                       v0Axis.y,
                       v0Axis.z),
           make_float3(minBoundsHit.x,
                       minBoundsHit.y,
                       minBoundsHit.z),
           __float_as_int(v0Axis.w),
           maxBoundsMiss.x, maxBoundsMiss.y,
           maxBoundsMiss.z, maxBoundsMiss.w,
           tempT, tempB1, tempB2);
        result |= hit;

        // XXX we could potentially merge t and tMax into a single word
        //     as they serve essentially the same purpose
        if(hit)
        {
          t = tempT;
          myInterval.y = tempT;
          tri = currentNode;
          b1 = tempB1;
          b2 = tempB2;
        } // end if

        // ensure that the miss and hit indices are the same
        // at this point
        maxBoundsMiss.w = minBoundsHit.w;
        hit = false;
      } // end else

      currentNode = hit ? __float_as_int(minBoundsHit.w) : __float_as_int(maxBoundsMiss.w);
    } // end while

    // write results
    stencil[i] = result;
    timeBarycentricsAndTriangleIndex[i] = make_float4(t, b1, b2, __int_as_float(tri));
  } // end if
} // end rbk()

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
                                    const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n/BLOCK_SIZE;

  if(gridSize)
    rbk<<<gridSize,BLOCK_SIZE>>>(NULL_NODE,
                                 rootIndex,
                                 origins,
                                 directions,
                                 interval,
                                 minBoundHitIndex,
                                 maxBoundMissIndex,
                                 firstVertexDominantAxis,
                                 stencil,
                                 timeBarycentricsAndTriangleIndex);
  if(n%BLOCK_SIZE)
    rbk<<<1,n%BLOCK_SIZE>>>(NULL_NODE,
                            rootIndex,
                            origins + gridSize*BLOCK_SIZE,
                            directions + gridSize*BLOCK_SIZE,
                            interval,
                            minBoundHitIndex,
                            maxBoundMissIndex,
                            firstVertexDominantAxis,
                            stencil + gridSize*BLOCK_SIZE,
                            timeBarycentricsAndTriangleIndex + gridSize*BLOCK_SIZE);
} // end cudaRayTriangleBVHIntersection()

__global__ void rbk(const unsigned int NULL_NODE,
                    const unsigned int rootIndex,
                    const float3 *origins,
                    const float3 *directions,
                    const float2 *intervals,
                    const float4 *minBoundHitIndex,
                    const float4 *maxBoundMissIndex,
                    const float4 *firstVertexDominantAxis,
                    bool *stencil,
                    float4 *timeBarycentricsAndTriangleIndex)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(stencil[i])
  {
    float3 origin = origins[i];
    float3 dir = directions[i];
    float2 interval = intervals[i];

    float3 invDir = make_float3(1.0f / dir.x,
                                1.0f / dir.y,
                                1.0f / dir.z);
    unsigned int currentNode = rootIndex;
    bool hit = false;
    bool result = false;
    float t = interval.y;;
    unsigned int tri = 0;
    float b1 = -1, b2 = -1;
    float4 minBoundsHit, maxBoundsMiss;
    float4 v0Axis;

    // XXX PERF: it might be possible to eliminate these temporaries
    float tempT, tempB1, tempB2;
    while(currentNode != NULL_NODE)
    {
      minBoundsHit = minBoundHitIndex[currentNode];
      maxBoundsMiss = maxBoundMissIndex[currentNode];

      // leaves (primitives) are listed before interior nodes
      // so bounding boxes occur after the root index
      if(currentNode >= rootIndex)
      {
        hit = intersectBox(origin,
                           invDir,
                           make_float3(minBoundsHit.x,
                                       minBoundsHit.y,
                                       minBoundsHit.z),
                           make_float3(maxBoundsMiss.x,
                                       maxBoundsMiss.y,
                                       maxBoundsMiss.z),
                           interval.x,
                           interval.y);
      } // end if
      else
      {
        v0Axis = firstVertexDominantAxis[currentNode];

        hit = cudaWaldBikkerIntersection
          (origin,
           dir,
           interval.x, interval.y,
           make_float3(v0Axis.x,
                       v0Axis.y,
                       v0Axis.z),
           make_float3(minBoundsHit.x,
                       minBoundsHit.y,
                       minBoundsHit.z),
           __float_as_int(v0Axis.w),
           maxBoundsMiss.x, maxBoundsMiss.y,
           maxBoundsMiss.z, maxBoundsMiss.w,
           tempT, tempB1, tempB2);
        result |= hit;

        // XXX we could potentially merge t and tMax into a single word
        //     as they serve essentially the same purpose
        if(hit)
        {
          t = tempT;
          interval.y = tempT;
          tri = currentNode;
          b1 = tempB1;
          b2 = tempB2;
        } // end if

        // ensure that the miss and hit indices are the same
        // at this point
        maxBoundsMiss.w = minBoundsHit.w;
        hit = false;
      } // end else

      currentNode = hit ? __float_as_int(minBoundsHit.w) : __float_as_int(maxBoundsMiss.w);
    } // end while

    // write results
    stencil[i] = result;
    timeBarycentricsAndTriangleIndex[i] = make_float4(t, b1, b2, __int_as_float(tri));
  } // end if
} // end rbk()

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
                                    const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n/BLOCK_SIZE;

  if(gridSize)
    rbk<<<gridSize,BLOCK_SIZE>>>(NULL_NODE,
                                 rootIndex,
                                 origins,
                                 directions,
                                 intervals,
                                 minBoundHitIndex,
                                 maxBoundMissIndex,
                                 firstVertexDominantAxis,
                                 stencil,
                                 timeBarycentricsAndTriangleIndex);
  if(n%BLOCK_SIZE)
    rbk<<<1,n%BLOCK_SIZE>>>(NULL_NODE,
                            rootIndex,
                            origins + gridSize*BLOCK_SIZE,
                            directions + gridSize*BLOCK_SIZE,
                            intervals + gridSize*BLOCK_SIZE,
                            minBoundHitIndex,
                            maxBoundMissIndex,
                            firstVertexDominantAxis,
                            stencil + gridSize*BLOCK_SIZE,
                            timeBarycentricsAndTriangleIndex + gridSize*BLOCK_SIZE);
} // end cudaRayTriangleBVHIntersection()

__global__ void shadowKernel(const unsigned int NULL_NODE,
                             const unsigned int rootIndex,
                             const float3 *rayOrigins,
                             const float3 *rayDirections,
                             const float2 *rayIntervals,
                             const float4 *minBoundHitIndex,
                             const float4 *maxBoundMissIndex,
                             const float4 *firstVertexDominantAxis,
                             const bool *stencil,
                             bool *results)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // threads which do no work are assumed to have their
  // rays blocked
  int result = 0;

  if(stencil[i])
  {
    // initially assume no triangle blocks the ray
    result = 1;

    float3 origin = rayOrigins[i];
    float3 dir = rayDirections[i];
    float2 interval = rayIntervals[i];

    float3 invDir = make_float3(1.0f / dir.x,
                                1.0f / dir.y,
                                1.0f / dir.z);
    unsigned int currentNode = rootIndex;
    bool hit = false;
    float4 minBoundsHit, maxBoundsMiss;
    float4 v0Axis;

    // XXX PERF: it might be possible to eliminate these temporaries
    float tempT, tempB1, tempB2;
    while(currentNode != NULL_NODE)
    {
      minBoundsHit = minBoundHitIndex[currentNode];
      maxBoundsMiss = maxBoundMissIndex[currentNode];

      // leaves (primitives) are listed before interior nodes
      // so bounding boxes occur after the root index
      if(currentNode >= rootIndex)
      {
        hit = intersectBox(origin,
                           invDir,
                           make_float3(minBoundsHit.x,
                                       minBoundsHit.y,
                                       minBoundsHit.z),
                           make_float3(maxBoundsMiss.x,
                                       maxBoundsMiss.y,
                                       maxBoundsMiss.z),
                           interval.x,
                           interval.y);
      } // end if
      else
      {
        v0Axis = firstVertexDominantAxis[currentNode];

        hit = cudaWaldBikkerIntersection
          (origin,
           dir,
           interval.x, interval.y,
           make_float3(v0Axis.x,
                       v0Axis.y,
                       v0Axis.z),
           make_float3(minBoundsHit.x,
                       minBoundsHit.y,
                       minBoundsHit.z),
           __float_as_int(v0Axis.w),
           maxBoundsMiss.x, maxBoundsMiss.y,
           maxBoundsMiss.z, maxBoundsMiss.w,
           tempT, tempB1, tempB2);

        if(hit)
        {
          // blocked
          result = 0;
          minBoundsHit.w = __int_as_float(NULL_NODE);
        } // end if

        // ensure that the miss and hit indices are the same
        // at this point
        maxBoundsMiss.w = minBoundsHit.w;
      } // end else

      currentNode = hit ? __float_as_int(minBoundsHit.w) : __float_as_int(maxBoundsMiss.w);
    } // end while
  } // end if

  // write results
  results[i] = result;
} // end shadowKernel()

void cudaShadowRayTriangleBVHIntersectionWithStencil(const unsigned int NULL_NODE,
                                                     const unsigned int rootIndex,
                                                     const float3* rayOrigins,
                                                     const float3* rayDirections,
                                                     const float2* rayIntervals,
                                                     const float4* minBoundHitIndex,
                                                     const float4* maxBoundMissIndex,
                                                     const float4* firstVertexDominantAxis,
                                                     const bool *stencil,
                                                     bool *results,
                                                     const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n/BLOCK_SIZE;

  if(gridSize)
    shadowKernel<<<gridSize,BLOCK_SIZE>>>(NULL_NODE,
                                          rootIndex,
                                          rayOrigins,
                                          rayDirections,
                                          rayIntervals,
                                          minBoundHitIndex,
                                          maxBoundMissIndex,
                                          firstVertexDominantAxis,
                                          stencil,
                                          results);
  if(n%BLOCK_SIZE)
    shadowKernel<<<1,n%BLOCK_SIZE>>>(NULL_NODE,
                                     rootIndex,
                                     rayOrigins + gridSize*BLOCK_SIZE,
                                     rayDirections + gridSize*BLOCK_SIZE,
                                     rayIntervals + gridSize*BLOCK_SIZE,
                                     minBoundHitIndex,
                                     maxBoundMissIndex,
                                     firstVertexDominantAxis,
                                     stencil + gridSize*BLOCK_SIZE,
                                     results + gridSize*BLOCK_SIZE);
} // end cudaRayTriangleBVHIntersection()

__global__ void shadowKernel(const unsigned int NULL_NODE,
                             const unsigned int rootIndex,
                             const float3 *rayOrigins,
                             const float3 *rayDirections,
                             const float2 rayInterval,
                             const float4 *minBoundHitIndex,
                             const float4 *maxBoundMissIndex,
                             const float4 *firstVertexDominantAxis,
                             const bool *stencil,
                             bool *results)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // threads which do no work are assumed to have their
  // rays blocked
  int result = 0;

  if(stencil[i])
  {
    // initially assume no triangle blocks the ray
    result = 1;

    float3 origin = rayOrigins[i];
    float3 dir = rayDirections[i];

    float3 invDir = make_float3(1.0f / dir.x,
                                1.0f / dir.y,
                                1.0f / dir.z);
    unsigned int currentNode = rootIndex;
    bool hit = false;
    float4 minBoundsHit, maxBoundsMiss;
    float4 v0Axis;

    // XXX PERF: it might be possible to eliminate these temporaries
    float tempT, tempB1, tempB2;
    while(currentNode != NULL_NODE)
    {
      minBoundsHit = minBoundHitIndex[currentNode];
      maxBoundsMiss = maxBoundMissIndex[currentNode];

      // leaves (primitives) are listed before interior nodes
      // so bounding boxes occur after the root index
      if(currentNode >= rootIndex)
      {
        hit = intersectBox(origin,
                           invDir,
                           make_float3(minBoundsHit.x,
                                       minBoundsHit.y,
                                       minBoundsHit.z),
                           make_float3(maxBoundsMiss.x,
                                       maxBoundsMiss.y,
                                       maxBoundsMiss.z),
                           rayInterval.x,
                           rayInterval.y);
      } // end if
      else
      {
        v0Axis = firstVertexDominantAxis[currentNode];

        hit = cudaWaldBikkerIntersection
          (origin,
           dir,
           rayInterval.x, rayInterval.y,
           make_float3(v0Axis.x,
                       v0Axis.y,
                       v0Axis.z),
           make_float3(minBoundsHit.x,
                       minBoundsHit.y,
                       minBoundsHit.z),
           __float_as_int(v0Axis.w),
           maxBoundsMiss.x, maxBoundsMiss.y,
           maxBoundsMiss.z, maxBoundsMiss.w,
           tempT, tempB1, tempB2);

        if(hit)
        {
          // blocked
          result = 0;
          minBoundsHit.w = __int_as_float(NULL_NODE);
        } // end if

        // ensure that the miss and hit indices are the same
        // at this point
        maxBoundsMiss.w = minBoundsHit.w;
      } // end else

      currentNode = hit ? __float_as_int(minBoundsHit.w) : __float_as_int(maxBoundsMiss.w);
    } // end while
  } // end if

  // write results
  results[i] = result;
} // end shadowKernel()

void cudaShadowRayTriangleBVHIntersectionWithStencil(const unsigned int NULL_NODE,
                                                     const unsigned int rootIndex,
                                                     const float3* rayOrigins,
                                                     const float3* rayDirections,
                                                     const float2 &rayInterval,
                                                     const float4* minBoundHitIndex,
                                                     const float4* maxBoundMissIndex,
                                                     const float4* firstVertexDominantAxis,
                                                     const bool *stencil,
                                                     bool *results,
                                                     const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n/BLOCK_SIZE;

  if(gridSize)
    shadowKernel<<<gridSize,BLOCK_SIZE>>>(NULL_NODE,
                                          rootIndex,
                                          rayOrigins,
                                          rayDirections,
                                          rayInterval,
                                          minBoundHitIndex,
                                          maxBoundMissIndex,
                                          firstVertexDominantAxis,
                                          stencil,
                                          results);
  if(n%BLOCK_SIZE)
    shadowKernel<<<1,n%BLOCK_SIZE>>>(NULL_NODE,
                                     rootIndex,
                                     rayOrigins + gridSize*BLOCK_SIZE,
                                     rayDirections + gridSize*BLOCK_SIZE,
                                     rayInterval,
                                     minBoundHitIndex,
                                     maxBoundMissIndex,
                                     firstVertexDominantAxis,
                                     stencil + gridSize*BLOCK_SIZE,
                                     results + gridSize*BLOCK_SIZE);
} // end cudaRayTriangleBVHIntersection()

