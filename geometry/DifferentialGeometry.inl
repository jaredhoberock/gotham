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
                         const float3 &dpdu, const float3 &dpdv,
                         const float3 &dndu, const float3 &dndv,
                         const ParametricCoordinates &uv, const Shape *s)
{
  setPoint(p);
  setPointPartials(dpdu, dpdv);
  setNormalVectorPartials(dndu, dndv);
  setParametricCoordinates(uv);
  setSurface(s);

  // compute the normal from its partials
  setNormalVector(dndu.cross(dndv).normalize());
} // end DifferentialGeometry::DifferentialGeometry()

DifferentialGeometry
  ::DifferentialGeometry(const Point &p, const Normal &n,
                         const float3 &dpdu, const float3 &dpdv,
                         const float3 &dndu, const float3 &dndv,
                         const ParametricCoordinates &uv, const Shape *s)
{
  setPoint(p);
  setNormalVector(n);
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

const Point &DifferentialGeometry
  ::getPoint(void) const
{
  return mPoint;
} // end DifferentialGeometry::getPoint()

void DifferentialGeometry
  ::setNormalVector(const Normal &n)
{
  mNormalVector = n;
} // end DifferentialGeometry::setNormalVector()

const Normal &DifferentialGeometry
  ::getNormalVector(void) const
{
  return mNormalVector;
} // end DifferentialGeometry::getNormalVector()

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

void DifferentialGeometry
  ::setSurface(const Shape *s)
{
  mSurface = s;
} // end DifferentialGeometry::setSurface()

const Shape *DifferentialGeometry
  ::getSurface(void) const
{
  return mSurface;
} // end DifferentialGeometry::getSurface()

void DifferentialGeometry
  ::setPointPartials(const float3 &dpdu, const float3 &dpdv)
{
  mPointPartials[0] = dpdu;
  mPointPartials[1] = dpdv;
} // end DifferentialGeometry::setPointPartials()

const float3 *DifferentialGeometry
  ::getPointPartials(void) const
{
  return mPointPartials;
} // end DifferentialGeometry::getPointPartials()

float3 *DifferentialGeometry
  ::getPointPartials(void)
{
  return mPointPartials;
} // end DifferentialGeometry::getPointPartials()

void DifferentialGeometry
  ::setNormalVectorPartials(const float3 &dndu, const float3 &dndv)
{
  mNormalVectorPartials[0] = dndu;
  mNormalVectorPartials[1] = dndv;
} // end DifferentialGeometry::setNormalVectorPartials()

const float3 *DifferentialGeometry
  ::getNormalVectorPartials(void) const
{
  return mNormalVectorPartials;
} // end DifferentialGeometry::getNormalVectorPartials()

float3 *DifferentialGeometry
  ::getNormalVectorPartials(void)
{ 
  return mNormalVectorPartials;
} // end DifferentialGeometry::getNormalVectorPartials()

