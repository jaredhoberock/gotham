/*! \file cudaCreateIntersections.cu
 *  \author Jared Hoberock
 *  \brief Implementation of cudaCreateIntersections function.
 */

#include "cudaCreateIntersections.h"
#include "../geometry/CudaDifferentialGeometry.h"
#include <stdcuda/vector_math.h>
#include <stdio.h>

// XXX we have to give these parameters stupid short names because of this bug
// http://forums.nvidia.com/index.php?showtopic=53767
// TODO fix this after the next CUDA release
__global__ void k(const float4 *o,
                  const float4 *d,
                  const float4 *tbi,         // hit time, barycentrics, and triangle index
                  const float3 *n,           // normals
                  const float2 *p0,          // parametrics for vertex 0
                  const float2 *p1,          // parametrics for vertex 1
                  const float2 *p2,          // parametrics for vertex 2
                  const float  *ia,          // inverse surface area per primitive
                  const PrimitiveHandle *ph, // primitive handles
                  const int    *s,           // stencil
                  CudaIntersection *in)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(s[i])
  {
    float4 org = o[i];
    float4 dir = d[i];
    float4 result = tbi[i];

    // compute hit point
    float3 p = make_float3(org.x,org.y,org.z) + result.x * make_float3(dir.x,dir.y,dir.z);
    float b0 = result.y;
    float b1 = result.z;
    float b2 = 1.0f - b1 - b0;

    // compute triangle index
    uint triIndex = __float_as_int(result.w);
    float3 ng  = n[triIndex];
    float2 uv0 = p0[triIndex];
    float2 uv1 = p1[triIndex];
    float2 uv2 = p2[triIndex];

    CudaDifferentialGeometry dg;

    // XXX compute partial derivatives

    // interpolate uv using barycentric coordinates
    float2 uv;
    uv.x = b0*uv0.x + b1*uv1.x + b2*uv2.x;
    uv.y = b0*uv0.y + b1*uv1.y + b2*uv2.y;

    dg.setPoint(p);
    dg.setNormal(ng);
    dg.setParametricCoordinates(uv);

    // XXX set tangent and binormal here

    // set the inverse surface area of the primitive
    float invA = ia[triIndex];
    dg.setInverseSurfaceArea(invA);

    // set the surface area of the primitive
    dg.setInverseSurfaceArea(1.0f / invA);

    // set the differential geometry
    in[i].setDifferentialGeometry(dg);

    // set the primitive handle
    //     in the intersection
    in[i].setPrimitive(ph[triIndex]);
  } // end if
} // end createIntersectionsKernel()

void cudaCreateIntersections(const float4 *rayOriginsAndMinT,
                             const float4 *rayDirectionsAndMaxT,
                             const float4 *timeBarycentricsAndTriangleIndex,
                             const float3 *geometricNormals,
                             const float2 *firstVertexParms,
                             const float2 *secondVertexParms,
                             const float2 *thirdVertexParms,
                             const float  *invSurfaceArea,
                             const PrimitiveHandle *primHandles,
                             const int *stencil,
                             CudaIntersection *intersections,
                             const size_t n)
{
  dim3 grid = dim3(1,1,1);
  dim3 block = dim3(n,1,1);

  k<<<grid,block>>>(rayOriginsAndMinT,
                    rayDirectionsAndMaxT,
                    timeBarycentricsAndTriangleIndex,
                    geometricNormals,
                    firstVertexParms,
                    secondVertexParms,
                    thirdVertexParms,
                    invSurfaceArea,
                    primHandles,
                    stencil,
                    intersections);
} // end cudaCreateIntersections()

