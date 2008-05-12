/*! \file CudaDifferentialGeometryVector.inl
 *  \author Jared Hoberock
 *  \brief Inline file for CudaDifferentialGeometryVector.h.
 */

#include "CudaDifferentialGeometryVector.h"

CudaDifferentialGeometryVector
  ::CudaDifferentialGeometryVector(const size_t n)
    :Parent()
{
  resize(n);
} // end CudaDifferentialGeometryVector::CudaDifferentialGeometryVector()

void CudaDifferentialGeometryVector
  ::resize(const size_t n)
{
  // resize each vector
  mPointsVector.resize(n);
  mNormalsVector.resize(n);
  mTangentsVector.resize(n);
  mBinormalsVector.resize(n);
  mParametricCoordinatesVector.resize(n);
  mSurfaceAreasVector.resize(n);
  mInverseSurfaceAreasVector.resize(n);
  mDPDUsVector.resize(n);
  mDPDVsVector.resize(n);
  mDNDUsVector.resize(n);
  mDNDVsVector.resize(n);

  // bind each pointer
  mPoints = &mPointsVector[0];
  mNormals = &mNormalsVector[0];
  mTangents = &mTangentsVector[0];
  mBinormals = &mBinormalsVector[0];
  mParametricCoordinates = &mParametricCoordinatesVector[0];
  mSurfaceAreas = &mSurfaceAreasVector[0];
  mInverseSurfaceAreas = &mInverseSurfaceAreasVector[0];
  mDPDUs = &mDPDUsVector[0];
  mDPDVs = &mDPDVsVector[0];
  mDNDUs = &mDNDUsVector[0];
  mDNDVs = &mDNDVsVector[0];
} // end CudaDifferentialGeometryVector::resize()

