/*! \file createDifferentialGeometry.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a function for
 *         creating a CudaDifferentialGeometry object
 *         given mesh data.
 */

#pragma once

#include "../geometry/CudaDifferentialGeometry.h"

/*! This function creates a CudaDifferentialGeometry object
 *  given a point, barycentrics, and triangle index, as well
 *  as the geometry description of a mesh.
 *  \param p The point of interest.
 *  \param b The barycentric coordinates of p in the triangle of interest.
 *  \param triIndex The index of the triangle of interest.
 *  \param v0 The entire list of first vertices of the triangles of the mesh.
 *  \param v1 The entire list of second vertices of the triangles of the mesh.
 *  \param v0 The entire list of third vertices of the triangles of the mesh.
 *  \param n The entire list of triangle normals.
 *  \param parms0 The entire list of first vertex parameterics of the triangles of the mesh.
 *  \param parms1 The entire list of second vertex parameterics of the triangles of the mesh.
 *  \param parms2 The entire list of third vertex parameterics of the triangles of the mesh.
 *  \param inverseSurfaceArea The inverse surface area list.
 *  \param dg The CudaDifferentialGeometry object will be returned here.
 */
inline __host__ __device__
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
                                  CudaDifferentialGeometry &dg);

#include "createDifferentialGeometry.inl"

