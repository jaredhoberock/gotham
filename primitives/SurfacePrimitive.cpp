/*! \file SurfacePrimitive.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SurfacePrimitive class.
 */

#include "SurfacePrimitive.h"
#include "../geometry/Ray.h"

SurfacePrimitive
  ::SurfacePrimitive(boost::shared_ptr<Surface> s,
                     const MaterialHandle &m)
    :mSurface(s),mMaterial(m)
{
  ;
} // end SurfacePrimitive::SurfacePrimitive()

SurfacePrimitive
  ::~SurfacePrimitive(void)
{
  ;
} // end SurfacePrimitive::SurfacePrimitive()

MaterialHandle SurfacePrimitive
  ::getMaterial(void) const
{
  return mMaterial;
} // end SurfacePrimitive::getMaterial()

void SurfacePrimitive
  ::getBoundingBox(BoundingBox &b) const
{
  mSurface->getBoundingBox(b);
} // end SurfacePrimitive::getBoundingBox()

bool SurfacePrimitive
  ::intersect(Ray &r, Intersection &inter) const
{
  float t;

  // XXX kill this const_cast
  DifferentialGeometry &dg = const_cast<DifferentialGeometry&>(inter.getDifferentialGeometry());
  if(!mSurface->intersect(r,t,dg))
  {
    return false;
  } // end if

  inter.setPrimitive(getPrimitiveHandle());

  // update the Ray's maximum t
  r.getInterval()[1] = t;

  return true;
} // end SurfacePrimitive::intersect()

bool SurfacePrimitive
  ::intersect(const Ray &r) const
{
  return mSurface->intersect(r);
} // end SurfacePrimitive::intersect()

const Surface *SurfacePrimitive
  ::getSurface(void) const
{
  return mSurface.get();
} // end SurfaceType::getSurface()

void SurfacePrimitive
  ::getSurface(boost::shared_ptr<Surface> &s) const
{
  s = mSurface;
} // end SurfacePrimitive::getSurface()

float SurfacePrimitive
  ::getSurfaceArea(void) const
{
  return mSurface->getSurfaceArea();
} // end SurfacePrimitive::getSurfaceArea()

float SurfacePrimitive
  ::getInverseSurfaceArea(void) const
{
  return mSurface->getInverseSurfaceArea();
} // end SurfacePrimitive::getInverseSurfaceArea()

void SurfacePrimitive
  ::sampleSurfaceArea(const float u0,
                      const float u1,
                      const float u2,
                      DifferentialGeometry &dg,
                      float &pdf) const
{
  mSurface->sampleSurfaceArea(u0,u1,u2,dg,pdf);
} // end SurfacePrimitive::sampleSurfaceArea()

float SurfacePrimitive
  ::evaluateSurfaceAreaPdf(const DifferentialGeometry &dg) const
{
  float result = mSurface->evaluateSurfaceAreaPdf(dg);
  return result;
} // end SurfacePrimitive::evaluatesurfaceAreaPdf()

void SurfacePrimitive
  ::setSurfacePrimitiveHandle(const PrimitiveHandle s)
{
  mSurfacePrimitiveHandle = s;
} // end SurfacePrimitive::setSurfacePrimitiveHandle()

PrimitiveHandle SurfacePrimitive
  ::getSurfacePrimitiveHandle(void) const
{
  return mSurfacePrimitiveHandle;
} // end SurfacePrimitive::getSurfacePrimitiveHandle()

