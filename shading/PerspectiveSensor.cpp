/*! \file PerspectiveSensor.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PerspectiveSensor class.
 */

#include "PerspectiveSensor.h"

PerspectiveSensor
  ::PerspectiveSensor(void)
{
  ;
} // end PerspectiveSensor::PerspectiveSensor()

PerspectiveSensor
  ::PerspectiveSensor(const float aspect,
                      const Point &origin,
                      const Vector3 &right,
                      const Vector3 &up)
    :mAspectRatio(aspect),
     mWindowOrigin(origin),
     mInverseWindowSurfaceArea(1.0f/(2.0f*2.0f*aspect)),
     mRight(right),
     mUp(up)
{
  ;
} // end PerspectiveSensor::PerspectiveSensor()

void PerspectiveSensor
  ::set(const float aspect,
        const Point &origin)
{
  mAspectRatio = aspect;
  mWindowOrigin = origin;
  mInverseWindowSurfaceArea = (1.0f/(2.0f*2.0f*mAspectRatio));
} // end PerspectiveSensor::set()

Spectrum PerspectiveSensor
  ::sample(const DifferentialGeometry &dg,
           const float u0,
           const float u1,
           const float u2,
           Vector3 &ws,
           float &pdf) const
{
  Point q;
  sampleWindow(u0,u1,
               mRight,
               mUp,
               dg.getNormal(),
               q,
               pdf);
  ws = q - dg.getPoint();
  float d2 = ws.norm2();
  ws /= sqrt(d2);

  // compute surface area pdf to solid angle pdf
  pdf *= d2;

  return Spectrum::white();
} // end PerspectiveSensor::sample()

void PerspectiveSensor
  ::sampleWindow(const float u,
                 const float v,
                 const Vector3 &xAxis,
                 const Vector3 &yAxis,
                 const Vector3 &zAxis,
                 Point &p,
                 float &pdf) const
{
  p = mWindowOrigin;
  p += 2.0f * mAspectRatio * u * xAxis;
  p += 2.0f * v * yAxis;
  pdf = mInverseWindowSurfaceArea;
} // end PerspectiveSensor::sampleWindow()

float PerspectiveSensor
  ::evaluatePdf(const Vector3 &ws,
                const DifferentialGeometry &dg) const
{
  // intersect a ray through dg in direction ws with the sensor window
  float t = dg.getNormal().dot(mWindowOrigin - dg.getPoint()) /
            dg.getNormal().dot(ws);

  // compute q the intersection with the ray and the window
  Point q = dg.getPoint() + t * ws;
  Point coords = q - mWindowOrigin;
  coords *= 0.5f;

  float u = coords.dot(dg.getPointPartials()[0]) / mAspectRatio;
  float v = coords.dot(dg.getPointPartials()[1]);

  // if the ray does not pass through the window,
  // then there is zero probability of having generated it
  if(u < 0.0f || u >= 1.0f) return 0.0f;
  if(v < 0.0f || v >= 1.0f) return 0.0f;

  // compute solid angle pdf
  Vector3 wi = q - dg.getPoint();
  float pdf = wi.norm2();
  pdf *= mInverseWindowSurfaceArea;
  return pdf;
} // end PerspectiveSensor::evaluatePdf()

Spectrum PerspectiveSensor
  ::evaluate(const Vector &ws,
             const DifferentialGeometry &dg) const
{
  // evaluate the pdf at ws, and if it is not zero, return white
  return evaluatePdf(ws, dg) > 0 ? Spectrum::white() : Spectrum::black();
} // end PerspectiveSensor::evaluate()

