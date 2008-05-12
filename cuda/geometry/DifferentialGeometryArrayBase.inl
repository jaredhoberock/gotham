/*! \file DifferentialGeometryArrayBase.inl
 *  \author Jared Hoberock
 *  \brief Inline file for DifferentialGeometryArrayBase.
 */

#include "DifferentialGeometryArrayBase.h"

template<typename P3, typename V3, typename P2, typename N3>
  DifferentialGeometryArrayBase<P3,V3,P2,N3> &DifferentialGeometryArrayBase<P3,V3,P2,N3>
    ::operator+=(const ptrdiff_t diff)
{
  // increment each pointer
  mPoints += diff;
  mNormals += diff;
  mTangents += diff;
  mBinormals += diff;
  mParametricCoordinates += diff;
  mSurfaceAreas += diff;
  mInverseSurfaceAreas += diff;
  mDPDUs += diff;
  mDPDVs += diff;
  mDNDUs += diff;
  mDNDVs += diff;

  return *this;
} // end CudaDifferentialGeometryArrayBase::operator+=()

template<typename P3, typename V3, typename P2, typename N3>
  DifferentialGeometryArrayBase<P3,V3,P2,N3> DifferentialGeometryArrayBase<P3,V3,P2,N3>
    ::operator+(const ptrdiff_t diff) const
{
  DifferentialGeometryArrayBase<P3,V3,P2,N3> result(*this);

  result += diff;
  
  return result;
} // end CudaDifferentialGeometryArrayBase::operator+=()

