/*! \file CudaTriangleList.cpp
 *  \author Jared Hoberock
 *  \brief C++ implementation of CudaTriangleList class.
 */

#include "CudaTriangleList.h"

void CudaTriangleList
  ::finalize(void)
{
  // build mSurfaceAreaPdf
  std::vector<unsigned int> triangleIndices;
  std::vector<float> triangleAreas;

  // collect triangle areas
  for(size_t i = 0; i != mFirstVertex.size(); ++i)
  {
    float3 temp0 = mFirstVertex[i];
    float3 temp1 = mSecondVertex[i];
    float3 temp2 = mThirdVertex[i];

    Vector v0(temp0.x, temp0.y, temp0.z);
    Vector v1(temp1.x, temp1.y, temp1.z);
    Vector v2(temp2.x, temp2.y, temp2.z);

    Vector e1 = v1 - v0;
    Vector e2 = v2 - v0;

    float a = 0.5f * e1.cross(e2).norm();

    triangleAreas.push_back(a);
    triangleIndices.push_back(i);
  } // end for i

  mSurfaceAreaPdf.build(triangleIndices.begin(), triangleIndices.end(),
                        triangleAreas.begin(), triangleAreas.end());
} // end CudaTriangleList::finalize()

