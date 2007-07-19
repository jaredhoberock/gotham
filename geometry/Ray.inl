/*! \file Ray.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Ray.h.
 */

#include <algorithm>

#include "Ray.h"

Ray
  ::Ray(void)
{
  getInterval()[0] = RAY_EPSILON;
  getInterval()[1] = RAY_INFINITY;
} // end Ray::Ray()

Ray
  ::Ray(const Point &a, const Vector3 &d)
    :mAnchor(a),mDirection(d),mInverseDirection(1.0f / d[0], 1.0f / d[1], 1.0f / d[2])
{
  getInterval()[0] = RAY_EPSILON;
  getInterval()[1] = RAY_INFINITY;
} // end Ray::Ray()

Ray
  ::Ray(const Point &a, const Vector3 &d, const float maxt)
    :mAnchor(a),mDirection(d),mInverseDirection(1.0f / d[0], 1.0f / d[1], 1.0f / d[2])
{
  getInterval()[0] = RAY_EPSILON;
  getInterval()[1] = maxt;
} // end Ray::Ray()

Ray
  ::Ray(const Point &a, const Vector3 &d, const float mint, const float maxt)
    :mAnchor(a),mDirection(d),mInverseDirection(1.0f / d[0], 1.0f / d[1], 1.0f / d[2])
{
  getInterval()[0] = std::min<float>(mint,maxt);
  getInterval()[1] = std::max<float>(mint,maxt);
} // end Ray::Ray()

const Point &Ray
  ::getAnchor(void) const
{
  return mAnchor;
} // end Ray::getAnchor()

void Ray
  ::setAnchor(const Point &a)
{
  mAnchor = a;
} // end Ray::setAnchor()

const Vector3 &Ray
  ::getDirection(void) const
{
  return mDirection;
} // end Ray::getDirection()

const Vector3 &Ray
  ::getInverseDirection(void) const
{
  return mInverseDirection;
} // end Ray::getInverseDirection()

void Ray
  ::setDirection(const Vector3 &d)
{
  mDirection = d;
  mInverseDirection[0] = 1.0f / d[0];
  mInverseDirection[1] = 1.0f / d[1];
  mInverseDirection[2] = 1.0f / d[2];
} // end Ray::setDirection()

Point Ray
  ::operator()(const float t) const
{
  return evaluate(t);
} // end Ray::operator()()

Point Ray
  ::evaluate(const float t) const
{
  return getAnchor() + t * getDirection();
} // end Ray::evaluate()

void Ray
  ::set(const Point &p0, const Point &p1)
{
  setAnchor(p0);
  setDirection(p1 - p0);
  getInterval()[0] = RAY_EPSILON;
  getInterval()[1] = 1.0f - RAY_EPSILON;
} // end Ray::set()

bool Ray
  ::contains(const float t) const
{
  return getInterval()[0] <= t && t <= getInterval()[1];
} // end Ray::contains()

const Ray::Interval &Ray
  ::getInterval(void) const
{
  return mInterval;
} // end Ray::getInterval()

Ray::Interval &Ray
  ::getInterval(void)
{
  return mInterval;
} // end Ray::getInterval()

