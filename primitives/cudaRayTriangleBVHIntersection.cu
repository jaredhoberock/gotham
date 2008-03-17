/*! \file cudaRayTriangleBVHIntersection.cu
 *  \author Jared Hoberock
 *  \brief Implementation of cudaRayTriangleBVHIntersection function.
 */

#include "cudaRayTriangleBVHIntersection.h"
#include <waldbikkerintersection/cudaWaldBikkerIntersection.h>
#include <stdcuda/vector_math.h>

#include <stdio.h>

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

__global__ void kernel(const unsigned int NULL_NODE,
                       const unsigned int rootIndex,
                       const float4 *rayOriginsAndMinT,
                       const float4 *rayDirectionsAndMaxT,
                       const float4 *minBoundHitIndex,
                       const float4 *maxBoundMissIndex,
                       const float4 *firstVertexDominantAxis,
                       int *stencil,
                       float4 *timeBarycentricsAndTriangleIndex)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(stencil[i])
  {
    float4 originAndMinT = rayOriginsAndMinT[i];
    float4 dirAndMaxT = rayDirectionsAndMaxT[i];
    printf("i: %d\n", i);
    printf("d: %f %f %f\n", dirAndMaxT.x, dirAndMaxT.y, dirAndMaxT.z);

    float3 invDir = make_float3(1.0f / dirAndMaxT.x,
                                1.0f / dirAndMaxT.y,
                                1.0f / dirAndMaxT.z);
    unsigned int currentNode = rootIndex;
    bool hit = false;
    bool result = false;
    float t = dirAndMaxT.w;
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

      //printf("currentNode: %u\n", currentNode);

      // leaves (primitives) are listed before interior nodes
      // so bounding boxes occur after the root index
      if(currentNode >= rootIndex)
      {
        hit = intersectBox(make_float3(originAndMinT.x,
                                       originAndMinT.y,
                                       originAndMinT.z),
                           invDir,
                           make_float3(minBoundsHit.x,
                                       minBoundsHit.y,
                                       minBoundsHit.z),
                           make_float3(maxBoundsMiss.x,
                                       maxBoundsMiss.y,
                                       maxBoundsMiss.z),
                           originAndMinT.w,
                           dirAndMaxT.w);
      } // end if
      else
      {
        printf("testing triangle\n");
        v0Axis = firstVertexDominantAxis[currentNode];
        hit = cudaWaldBikkerIntersection
          (make_float3(originAndMinT.x, originAndMinT.y, originAndMinT.z),
           make_float3(dirAndMaxT.x, dirAndMaxT.y, dirAndMaxT.z),
           originAndMinT.w, dirAndMaxT.w,
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
          dirAndMaxT.w = tempT;
          tri = currentNode;
          b1 = tempB1;
          b2 = tempB2;
        } // end if

        // ensure that the miss and hit indices are the same
        // at this point
        maxBoundsMiss.w = minBoundsHit.w;
      } // end else

      currentNode = hit ? __float_as_int(minBoundsHit.w) : __float_as_int(maxBoundsMiss.w);
    } // end while
    
    // write results
    stencil[i] = result;
    timeBarycentricsAndTriangleIndex[i] = make_float4(t, b1, b2, __int_as_float(tri));
  } // end if

  //stencil[i] = 0;
} // end kernel()

void cudaRayTriangleBVHIntersection(const unsigned int NULL_NODE,
                                    const unsigned int rootIndex,
                                    const float4* rayOriginsAndMinT,
                                    const float4* rayDirectionsAndMaxT,
                                    const float4* minBoundHitIndex,
                                    const float4* maxBoundMissIndex,
                                    const float4* firstVertexDominantAxis,
                                    int* stencil,
                                    float4* timeBarycentricsAndTriangleIndex,
                                    const size_t n)
{
  dim3 grid = dim3(1,1,1);
  dim3 block = dim3(n,1,1);

  kernel<<<grid,block>>>(NULL_NODE,
                         rootIndex,
                         rayOriginsAndMinT,
                         rayDirectionsAndMaxT,
                         minBoundHitIndex,
                         maxBoundMissIndex,
                         firstVertexDominantAxis,
                         stencil,
                         timeBarycentricsAndTriangleIndex);
} // end cudaRayTriangleBVHIntersection()
