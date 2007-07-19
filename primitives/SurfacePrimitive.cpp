/*! \file SurfacePrimitive.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SurfacePrimitive class.
 */

#include "SurfacePrimitive.h"
#include "../geometry/Ray.h"

SurfacePrimitive
  ::SurfacePrimitive(boost::shared_ptr<Surface> s,
                     boost::shared_ptr<Material> m)
    :mSurface(s),mMaterial(m)
{
  ;
} // end SurfacePrimitive::SurfacePrimitive()

const Material *SurfacePrimitive
  ::getMaterial(void) const
{
  return mMaterial.get();
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

  inter.setPrimitive(this);

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

float SurfacePrimitive
  ::getSurfaceArea(void) const
{
  return mSurface->getSurfaceArea();
} // end SurfacePrimitive::getSurfaceArea()

void SurfacePrimitive
  ::sampleSurfaceArea(const float u0,
                      const float u1,
                      const float u2,
                      DifferentialGeometry &dg,
                      float &pdf) const
{
  return mSurface->sampleSurfaceArea(u0,u1,u2,dg,pdf);
} // end SurfacePrimitive::sampleSurfaceArea()

