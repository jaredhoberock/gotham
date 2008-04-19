/*! \file createDifferentialGeometry.inl
 *  \author Jared Hoberock
 *  \brief Inline file for createDifferentialGeometry.h.
 */

#include "createDifferentialGeometry.h"
#include <stdcuda/vector_math.h>

void createDifferentialGeometry(const float3 &p,
                                const float2 &b,
                                const unsigned int triIndex,
                                const float3 *v0,
                                const float3 *v1,
                                const float3 *v2,
                                const float3 *n,
                                const float2 *parms0,
                                const float2 *parms1,
                                const float2 *parms2,
                                const float *inverseSurfaceArea,
                                CudaDifferentialGeometry &dg)
{
  float3 ng  = n[triIndex];
  float2 uv0 = parms0[triIndex];
  float2 uv1 = parms1[triIndex];
  float2 uv2 = parms2[triIndex];

  // XXX compute partial derivatives
  // compute deltas for partial derivatives
  float du1 = uv0.x - uv2.x;
  float du2 = uv1.x - uv2.x;
  float dv1 = uv0.y - uv2.y;
  float dv2 = uv1.y - uv2.y;
  float3 dp1 = v0[triIndex] - v2[triIndex], 
         dp2 = v1[triIndex] - v2[triIndex];
  float determinant = du1 * dv2 - dv1 * du2;
  float invDet = 1.0f / determinant;

  float3 dpdu = ( dv2*dp1 - dv1*dp2) * invDet;
  float3 dpdv = (-du2*dp1 + du1*dp2) * invDet;

  dg.setPointPartials(dpdu,dpdv);

  // interpolate uv using barycentric coordinates
  float2 uv;
  float b2 = 1.0f - b.y - b.x;
  uv.x = b.x*uv0.x + b.y*uv1.x + b2*uv2.x;
  uv.y = b.x*uv0.y + b.y*uv1.y + b2*uv2.y;

  dg.setPoint(p);
  dg.setNormal(ng);
  dg.setParametricCoordinates(uv);

  float3 tangent = normalize(dpdu);
  dg.setTangent(tangent);

  // force an orthonormal basis
  dg.setBinormal(cross(ng, tangent));

  // set the inverse surface area of the primitive
  float invA = inverseSurfaceArea[triIndex];
  dg.setInverseSurfaceArea(invA);

  // set the surface area of the primitive
  dg.setSurfaceArea(1.0f / invA);
} // end createDifferentialGeometry()

