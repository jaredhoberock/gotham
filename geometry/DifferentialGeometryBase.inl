/*! \file DifferentialGeometry.inl
 *  \author Jared Hoberock
 *  \brief Inline file for DifferentialGeometry.h.
 */

#include "DifferentialGeometry.h"

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  DifferentialGeometryBase<P3,V3,P2,N3>
    ::DifferentialGeometryBase(void)
{
  ;
} // end DifferentialGeometryBase::DifferentialGeometryBase()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  DifferentialGeometryBase<P3,V3,P2,N3>
    ::DifferentialGeometryBase(const Point &p,
                               const Vector &dpdu, const Vector &dpdv,
                               const Vector &dndu, const Vector &dndv,
                               const ParametricCoordinates &uv,
                               const float a,
                               const float invA)
{
  setPoint(p);
  setPointPartials(dpdu, dpdv);
  setNormalVectorPartials(dndu, dndv);
  setParametricCoordinates(uv);
  setSurfaceArea(a);
  setInverseSurfaceArea(invA);

  // compute the normal from its partials
  setNormal(dndu.cross(dndv).normalize());
} // end DifferentialGeometryBase::DifferentialGeometryBase()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  DifferentialGeometryBase<P3,V3,P2,N3>
    ::DifferentialGeometryBase(const Point &p, const Normal &n,
                               const Vector &dpdu, const Vector &dpdv,
                               const Vector &dndu, const Vector &dndv,
                               const ParametricCoordinates &uv,
                               const float a,
                               const float invA)
{
  setPoint(p);
  setNormal(n);
  setPointPartials(dpdu, dpdv);
  setNormalVectorPartials(dndu, dndv);
  setParametricCoordinates(uv);
  setSurfaceArea(a);
  setInverseSurfaceArea(invA);
} // end DifferentialGeometryBase::DifferentialGeometryBase()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  void DifferentialGeometryBase<P3,V3,P2,N3>
    ::setPoint(const Point &p)
{
  mPoint = p;
} // end DifferentialGeometryBase::setPoint()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  typename DifferentialGeometryBase<P3,V3,P2,N3>::Point &DifferentialGeometryBase<P3,V3,P2,N3>
    ::getPoint(void)
{
  return mPoint;
} // end DifferentialGeometryBase::getPoint()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  const typename DifferentialGeometryBase<P3,V3,P2,N3>::Point &DifferentialGeometryBase<P3,V3,P2,N3>
    ::getPoint(void) const
{
  return mPoint;
} // end DifferentialGeometryBase::getPoint()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  void DifferentialGeometryBase<P3,V3,P2,N3>
    ::setNormal(const Normal &n)
{
  mNormal= n;
} // end DifferentialGeometryBase::setNormal()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  typename DifferentialGeometryBase<P3,V3,P2,N3>::Normal &DifferentialGeometryBase<P3,V3,P2,N3>
    ::getNormal(void)
{
  return mNormal;
} // end DifferentialGeometryBase::getNormal()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  const typename DifferentialGeometryBase<P3,V3,P2,N3>::Normal &DifferentialGeometryBase<P3,V3,P2,N3>
    ::getNormal(void) const
{
  return mNormal;
} // end DifferentialGeometryBase::getNormal()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  void DifferentialGeometryBase<P3,V3,P2,N3>
    ::setParametricCoordinates(const ParametricCoordinates &uv)
{
  mParametricCoordinates = uv;
} // end DifferentialGeometryBase::setParametricCoordinates()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  const typename DifferentialGeometryBase<P3,V3,P2,N3>::ParametricCoordinates &DifferentialGeometryBase<P3,V3,P2,N3>
    ::getParametricCoordinates(void) const
{
  return mParametricCoordinates;
} // end DifferentialGeometryBase::getParametricCoordinates()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  typename DifferentialGeometryBase<P3,V3,P2,N3>::ParametricCoordinates &DifferentialGeometryBase<P3,V3,P2,N3>
    ::getParametricCoordinates(void)
{
  return mParametricCoordinates;
} // end DifferentialGeometryBase::getParametricCoordinates()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  void DifferentialGeometryBase<P3,V3,P2,N3>
    ::setSurfaceArea(const float a)
{
  mSurfaceArea = a;
} // end DifferentialGeometryBase::setSurfaceArea()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  float DifferentialGeometryBase<P3,V3,P2,N3>
    ::getSurfaceArea(void) const
{
  return mSurfaceArea;
} // end DifferentialGeometryBase::getSurfaceArea()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  void DifferentialGeometryBase<P3,V3,P2,N3>
    ::setInverseSurfaceArea(const float invA)
{
  mInverseSurfaceArea = invA;
} // end DifferentialGeometryBase::setInverseSurfaceArea()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  float DifferentialGeometryBase<P3,V3,P2,N3>
    ::getInverseSurfaceArea(void) const
{
  return mInverseSurfaceArea;
} // end DifferentialGeometryBase::getInverseSurfaceArea()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  void DifferentialGeometryBase<P3,V3,P2,N3>
    ::setPointPartials(const Vector &dpdu, const Vector &dpdv)
{
  mDPDU = dpdu;
  mDPDV = dpdv;
} // end DifferentialGeometryBase::setPointPartials()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  void DifferentialGeometryBase<P3,V3,P2,N3>
    ::setNormalVectorPartials(const Vector &dndu, const Vector &dndv)
{
  mNormalVectorPartials[0] = dndu;
  mNormalVectorPartials[1] = dndv;
} // end DifferentialGeometryBase::setNormalVectorPartials()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  const typename DifferentialGeometryBase<P3,V3,P2,N3>::Vector *DifferentialGeometryBase<P3,V3,P2,N3>
    ::getNormalVectorPartials(void) const
{
  return mNormalVectorPartials;
} // end DifferentialGeometryBase::getNormalVectorPartials()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  typename DifferentialGeometryBase<P3,V3,P2,N3>::Vector *DifferentialGeometryBase<P3,V3,P2,N3>
    ::getNormalVectorPartials(void)
{ 
  return mNormalVectorPartials;
} // end DifferentialGeometryBase::getNormalVectorPartials()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  void DifferentialGeometryBase<P3,V3,P2,N3>
    ::setTangent(const Vector &t)
{
  mTangent = t;
} // end DifferentialGeometryBase::setTangent()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  const typename DifferentialGeometryBase<P3,V3,P2,N3>::Vector &DifferentialGeometryBase<P3,V3,P2,N3>
    ::getTangent(void) const
{
  return mTangent;
} // end DifferentialGeometryBase::getTangent()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  void DifferentialGeometryBase<P3,V3,P2,N3>
    ::setBinormal(const Vector &b)
{
  mBinormal = b;
} // end DifferentialGeometryBase::setBinormal()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  const typename DifferentialGeometryBase<P3,V3,P2,N3>::Vector &DifferentialGeometryBase<P3,V3,P2,N3>
    ::getBinormal(void) const
{
  return mBinormal;
} // end DifferentialGeometryBase::getBinormal()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  const V3 &DifferentialGeometryBase<P3,V3,P2,N3>
    ::getDPDU(void) const
{
  return mDPDU;
} // end DifferentialGeometryBase::getDPDU()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  const V3 &DifferentialGeometryBase<P3,V3,P2,N3>
    ::getDPDV(void) const
{
  return mDPDV;
} // end DifferentialGeometryBase::getDPDV()

