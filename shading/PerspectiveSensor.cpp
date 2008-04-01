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
  ::PerspectiveSensor(const Spectrum &response,
                      const float aspect,
                      const Point &origin,
                      const Vector3 &right,
                      const Vector3 &up)
    :mAspectRatio(aspect),
     mInverseWindowSurfaceArea(1.0f/(2.0f*2.0f*aspect)),
     mWindowOrigin(origin),
     mRight(right),
     mUp(up),
     mResponse(response)
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
           float &pdf,
           bool &delta) const
{
  delta = false;
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

  // this removes the vignetting effect, but is it correct?
  pdf *= dg.getNormal().dot(ws);

  //// divide the response by the dot product to remove the vignette effect
  //return mResponse / dg.getNormal().dot(ws);
  return mResponse;
} // end PerspectiveSensor::sample()

void PerspectiveSensor
  ::invert(const Vector &w,
           const DifferentialGeometry &dg,
           float &u0,
           float &u1) const
{
  // a ray from the dg in direction w intersects the window at
  // time t
  // remember that the normal points in the -look direction
  float t =
    -dg.getNormal().dot(mWindowOrigin - dg.getPoint()) /
    -dg.getNormal().dot(w);

  // compute q the intersection with the ray and the plane
  Point q = dg.getPoint() + t * w;

  // this is the inverse operation of sampleFilmPlane():
  // get the film plane coordinates of q
  q -= mWindowOrigin;
  q *= 0.5f;

  u0 = q.dot(mRight) / mAspectRatio;
  u1 = q.dot(mUp);
} // end PerspectiveSensor::invert()

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
  // remember that the normal points in the -look direction
  float t = -dg.getNormal().dot(mWindowOrigin - dg.getPoint()) /
            -dg.getNormal().dot(ws);

  // if t is negative, then ws came from 'behind the camera',
  // and there is zero pdf of generating such directions
  if(t < 0) return 0;

  // compute q the intersection with the ray and the window
  Point q = dg.getPoint() + t * ws;
  Point coords = q - mWindowOrigin;
  coords *= 0.5f;

  float u = coords.dot(mRight) / mAspectRatio;
  float v = coords.dot(mUp);

  // if the ray does not pass through the window,
  // then there is zero probability of having generated it
  if(u < 0.0f || u >= 1.0f) return 0.0f;
  if(v < 0.0f || v >= 1.0f) return 0.0f;

  // compute solid angle pdf
  Vector3 wi = q - dg.getPoint();
  float pdf = wi.norm2();
  pdf *= mInverseWindowSurfaceArea;

  // this removes the vignetting effect, but is it correct?
  pdf *= dg.getNormal().absDot(ws);

  return pdf;
} // end PerspectiveSensor::evaluatePdf()

Spectrum PerspectiveSensor
  ::evaluate(const Vector &ws,
             const DifferentialGeometry &dg) const
{
  // evaluate the pdf at ws, and if it is not zero, return mResponse
  // divide by dot product to remove the vignetting effect
  //return evaluatePdf(ws, dg) > 0 ? (mResponse / dg.getNormal().absDot(ws)) : Spectrum::black();
  Spectrum result = evaluatePdf(ws, dg) > 0 ? (mResponse) : Spectrum::black();

  return result;
} // end PerspectiveSensor::evaluate()

Vector PerspectiveSensor
  ::getRight(void) const
{
  return mRight;
} // end PerspectiveSensor::getRight()

Vector PerspectiveSensor
  ::getUp(void) const
{
  return mUp;
} // end PerspectiveSensor::getUp()

