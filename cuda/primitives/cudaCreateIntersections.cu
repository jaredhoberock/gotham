/*! \file cudaCreateIntersections.cu
 *  \author Jared Hoberock
 *  \brief Implementation of cudaCreateIntersections function.
 */

#include "cudaCreateIntersections.h"
#include "../geometry/CudaDifferentialGeometry.h"
#include "createDifferentialGeometry.h"
#include <stdcuda/vector_math.h>
#include <stdio.h>

struct Parameters
{
  const float4 *o;
  const float4 *d;
  const float4 *hitTimeBarycentricsTriangleIndex;
  const float3 *n;
  const float3 *v0;
  const float3 *v1;
  const float3 *v2;
  const float2 *uv0;
  const float2 *uv1;
  const float2 *uv2;
  const float  *inverseSurfaceArea;
  const PrimitiveHandle *primitiveHandles;
  const bool   *stencil;
  CudaIntersection *results;
};

// XXX we have to give these parameters stupid short names because of this bug
// http://forums.nvidia.com/index.php?showtopic=53767
// TODO fix this after the next CUDA release
__global__ void k(const Parameters p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(p.stencil[i])
  {
    float4 org = p.o[i];
    float4 dir = p.d[i];
    float4 result = p.hitTimeBarycentricsTriangleIndex[i];

    // compute hit point
    float3 hitPoint = make_float3(org.x,org.y,org.z) + result.x * make_float3(dir.x,dir.y,dir.z);

    // compute triangle index
    uint triIndex = __float_as_int(result.w);

    CudaDifferentialGeometry dg;
    createDifferentialGeometry(hitPoint, make_float2(result.y,result.z),
                               triIndex,
                               p.v0, p.v1, p.v2, p.n,
                               p.uv0, p.uv1, p.uv2,
                               p.inverseSurfaceArea,
                               dg);

    // set the DifferentialGeometry
    p.results[i].setDifferentialGeometry(dg);

    // set the primitive handle
    //     in the intersection
    p.results[i].setPrimitive(p.primitiveHandles[triIndex]);
  } // end if
} // end createIntersectionsKernel()

//// XXX we have to give these parameters stupid short names because of this bug
//// http://forums.nvidia.com/index.php?showtopic=53767
//// TODO fix this after the next CUDA release
//__global__ void k(const float4 *o,
//                  const float4 *d,
//                  const float4 *tbi,         // hit time, barycentrics, and triangle index
//                  const float3 *n,           // normals
//                  const float3 *v0,          // position for vertex 0
//                  const float3 *v1,          // position for vertex 1
//                  const float3 *v2,          // position for vertex 2
//                  const float2 *p0,          // parametrics for vertex 0
//                  const float2 *p1,          // parametrics for vertex 1
//                  const float2 *p2,          // parametrics for vertex 2
//                  const float  *ia,          // inverse surface area per primitive
//                  const PrimitiveHandle *h,  // primitive handles
//                  const bool    *s,          // stencil
//                  CudaIntersection *in)
//{
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
//
//  if(s[i])
//  {
//    float4 org = o[i];
//    float4 dir = d[i];
//    float4 result = tbi[i];
//
//    // compute hit point
//    float3 p = make_float3(org.x,org.y,org.z) + result.x * make_float3(dir.x,dir.y,dir.z);
//    float b0 = result.y;
//    float b1 = result.z;
//    float b2 = 1.0f - b1 - b0;
//
//    // compute triangle index
//    uint triIndex = __float_as_int(result.w);
//    float3 ng  = n[triIndex];
//    float2 uv0 = p0[triIndex];
//    float2 uv1 = p1[triIndex];
//    float2 uv2 = p2[triIndex];
//
//    CudaDifferentialGeometry &dg = in[i].getDifferentialGeometry();
//
//    // XXX compute partial derivatives
//    // compute deltas for partial derivatives
//    float du1 = uv0.x - uv2.x;
//    float du2 = uv1.x - uv2.x;
//    float dv1 = uv0.y - uv2.y;
//    float dv2 = uv1.y - uv2.y;
//    float3 dp1 = v0[triIndex] - v2[triIndex], dp2 = v1[triIndex] - v2[triIndex];
//    float determinant = du1 * dv2 - dv1 * du2;
//    if(determinant == 0)
//    {
//      // handle zero determinant case
//    } // end if
//    else
//    {
//      float invDet = 1.0f / determinant;
//      dg.getPointPartials()[0] = ( dv2*dp1 - dv1*dp2) * invDet;
//      dg.getPointPartials()[1] = (-du2*dp1 + du1*dp2) * invDet;
//    } // end else
//
//    // interpolate uv using barycentric coordinates
//    float2 uv;
//    uv.x = b0*uv0.x + b1*uv1.x + b2*uv2.x;
//    uv.y = b0*uv0.y + b1*uv1.y + b2*uv2.y;
//
//    dg.setPoint(p);
//    dg.setNormal(ng);
//    dg.setParametricCoordinates(uv);
//    dg.setTangent(normalize(dg.getPointPartials()[0]));
//
//    // force an orthonormal basis
//    dg.setBinormal(cross(ng, dg.getTangent()));
//
//    // XXX set tangent and binormal here
//
//    // set the inverse surface area of the primitive
//    float invA = ia[triIndex];
//    dg.setInverseSurfaceArea(invA);
//
//    // set the surface area of the primitive
//    dg.setInverseSurfaceArea(1.0f / invA);
//
//    // set the primitive handle
//    //     in the intersection
//    in[i].setPrimitive(h[triIndex]);
//  } // end if
//} // end createIntersectionsKernel()

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
                             const size_t n)
{
  unsigned int BLOCK_SIZE = 192;
  unsigned int gridSize = n / 192;

  Parameters p = {rayOriginsAndMinT,
                  rayDirectionsAndMaxT,
                  timeBarycentricsAndTriangleIndex,
                  geometricNormals,
                  firstVertex,
                  secondVertex,
                  thirdVertex,
                  firstVertexParms,
                  secondVertexParms,
                  thirdVertexParms,
                  invSurfaceArea,
                  primHandles,
                  stencil,
                  intersections};

  if(gridSize)
    k<<<gridSize,BLOCK_SIZE>>>(p);
  if(n%BLOCK_SIZE)
  {
    p.o += gridSize*BLOCK_SIZE;
    p.d += gridSize*BLOCK_SIZE;
    p.hitTimeBarycentricsTriangleIndex += gridSize*BLOCK_SIZE;
    p.stencil += gridSize*BLOCK_SIZE;
    p.results += gridSize*BLOCK_SIZE;

    k<<<1,n%BLOCK_SIZE>>>(p);
  } // end if
} // end cudaCreateIntersections()

