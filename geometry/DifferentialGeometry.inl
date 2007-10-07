/*! \file DifferentialGeometry.inl
 *  \author Jared Hoberock
 *  \brief Inline file for DifferentialGeometry.h.
 */

#include "DifferentialGeometry.h"

DifferentialGeometry
  ::DifferentialGeometry(void)
{
  ;
} // end DifferentialGeometry::DifferentialGeometry()

DifferentialGeometry
  ::DifferentialGeometry(const Point &p,
                         const Vector3 &dpdu, const Vector3 &dpdv,
                         const Vector3 &dndu, const Vector3 &dndv,
                         const ParametricCoordinates &uv,
                         const SurfacePrimitive *s)
{
  setPoint(p);
  setPointPartials(dpdu, dpdv);
  setNormalVectorPartials(dndu, dndv);
  setParametricCoordinates(uv);
  setSurface(s);

  // compute the normal from its partials
  setNormal(dndu.cross(dndv).normalize());
} // end DifferentialGeometry::DifferentialGeometry()

DifferentialGeometry
  ::DifferentialGeometry(const Point &p, const Normal &n,
                         const Vector3 &dpdu, const Vector3 &dpdv,
                         const Vector3 &dndu, const Vector3 &dndv,
                         const ParametricCoordinates &uv,
                         const SurfacePrimitive *s)
{
  setPoint(p);
  setNormal(n);
  setPointPartials(dpdu, dpdv);
  setNormalVectorPartials(dndu, dndv);
  setParametricCoordinates(uv);
  setSurface(s);
} // end DifferentialGeometry::DifferentialGeometry()

void DifferentialGeometry
  ::setPoint(const Point &p)
{
  mPoint = p;
} // end DifferentialGeometry::setPoint()

Point &DifferentialGeometry
  ::getPoint(void)
{
  return mPoint;
} // end DifferentialGeometry::getPoint()

const Point &DifferentialGeometry
  ::getPoint(void) const
{
  return mPoint;
} // end DifferentialGeometry::getPoint()

void DifferentialGeometry
  ::setNormal(const Normal &n)
{
  mNormal= n;
} // end DifferentialGeometry::setNormal()

Normal &DifferentialGeometry
  ::getNormal(void)
{
  return mNormal;
} // end DifferentialGeometry::getNormal()

const Normal &DifferentialGeometry
  ::getNormal(void) const
{
  return mNormal;
} // end DifferentialGeometry::getNormal()

void DifferentialGeometry
  ::setParametricCoordinates(const ParametricCoordinates &uv)
{
  mParametricCoordinates = uv;
} // end DifferentialGeometry::setParametricCoordinates()

const ParametricCoordinates &DifferentialGeometry
  ::getParametricCoordinates(void) const
{
  return mParametricCoordinates;
} // end DifferentialGeometry::getParametricCoordinates()

ParametricCoordinates &DifferentialGeometry
  ::getParametricCoordinates(void)
{
  return mParametricCoordinates;
} // end DifferentialGeometry::getParametricCoordinates()

void DifferentialGeometry
  ::setSurface(const SurfacePrimitive *s)
{
  mSurface = s;
} // end DifferentialGeometry::setSurface()

const SurfacePrimitive *DifferentialGeometry
  ::getSurface(void) const
{
  return mSurface;
} // end DifferentialGeometry::getSurface()

void DifferentialGeometry
  ::setPointPartials(const Vector3 &dpdu, const Vector3 &dpdv)
{
  mPointPartials[0] = dpdu;
  mPointPartials[1] = dpdv;
} // end DifferentialGeometry::setPointPartials()

const Vector3 *DifferentialGeometry
  ::getPointPartials(void) const
{
  return mPointPartials;
} // end DifferentialGeometry::getPointPartials()

Vector3 *DifferentialGeometry
  ::getPointPartials(void)
{
  return mPointPartials;
} // end DifferentialGeometry::getPointPartials()

void DifferentialGeometry
  ::setNormalVectorPartials(const Vector3 &dndu, const Vector3 &dndv)
{
  mNormalVectorPartials[0] = dndu;
  mNormalVectorPartials[1] = dndv;
} // end DifferentialGeometry::setNormalVectorPartials()

const Vector3 *DifferentialGeometry
  ::getNormalVectorPartials(void) const
{
  return mNormalVectorPartials;
} // end DifferentialGeometry::getNormalVectorPartials()

Vector3 *DifferentialGeometry
  ::getNormalVectorPartials(void)
{ 
  return mNormalVectorPartials;
} // end DifferentialGeometry::getNormalVectorPartials()

void DifferentialGeometry
  ::setTangent(const Vector &t)
{
  mTangent = t;
} // end DifferentialGeometry::setTangent()

const Vector &DifferentialGeometry
  ::getTangent(void) const
{
  return mTangent;
} // end DifferentialGeometry::getTangent()

void DifferentialGeometry
  ::setBinormal(const Vector &b)
{
  mBinormal = b;
} // end DifferentialGeometry::setBinormal()

const Vector &DifferentialGeometry
  ::getBinormal(void) const
{
  return mBinormal;
} // end DifferentialGeometry::getBinormal()


