/*! \file BoundingBox.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of BoundingBox class.
 */

#include "BoundingBox.h"
#include "Ray.h"
#include "Normal.h"

bool BoundingBox
  ::intersect(const Ray &r, float &t0, float &t1) const
{
  t0 = r.getInterval()[0];
  t1 = r.getInterval()[1];

  const Vector &invRayDir = r.getInverseDirection();

  // we use the slabs method
  float tNear, tFar;
  for(int i = 0; i < 3; ++i)
  {
    // update the interval for the ith bounding box slab
    tNear = (getMinCorner()[i] - r.getAnchor()[i]) * invRayDir[i];
    tFar = (getMaxCorner()[i] - r.getAnchor()[i]) * invRayDir[i];

    // update parametric interval from slab intersection times
    if(tNear > tFar) std::swap(tNear, tFar);
    t0 = (tNear > t0)?(tNear):(t0);
    t1 = (tFar < t1)?(tFar):(t1);

    if(t0 > t1) return false;
  } // end for i

  return true;
} // end BoundingBox::intersect()

bool BoundingBox
  ::intersect(const Ray &r, float &t, Normal &n) const
{
  float t0 = r.getInterval()[0];
  float t1 = r.getInterval()[1];

  // we use the slabs method
  bool flipNormal = false;
  float invRayDir, tNear, tFar;
  for(int i = 0; i < 3; ++i)
  {
    // update the interval for the ith bounding box slab
    invRayDir = 1.0f / r.getDirection()[i];
    tNear = (getMinCorner()[i] - r.getAnchor()[i]) * invRayDir;
    tFar = (getMaxCorner()[i] - r.getAnchor()[i]) * invRayDir;

    // update parametric interval from slab intersection times
    flipNormal = true;
    if(tNear > tFar)
    {
      flipNormal = false;
      std::swap(tNear, tFar);
    } // end if

    if(tNear > t0)
    {
      // update normal
      n[0] = n[1] = n[2] = 0.0f;
      n[i] = (flipNormal)?(-1.0f):(1.0f);
      t0 = tNear;
    } // end if

    t1 = (tFar < t1)?(tFar):(t1);

    if(t0 > t1) return false;
  } // end for i

  t = t0;

  return true;
} // end BoundingBox::intersect()

