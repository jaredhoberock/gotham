/*! \file PerspectiveSensorBase.inl
 *  \author Jared Hoberock
 *  \brief Inline file for PerspectiveSensorBase.h.
 */

#include "PerspectiveSensorBase.h"

template<typename V3, typename S3>
  PerspectiveSensorBase<V3,S3>
    ::PerspectiveSensorBase(void)
{
  ;
} // end PerspectiveSensorBase::PerspectiveSensorBase()

template<typename V3, typename S3>
  PerspectiveSensorBase<V3,S3>
    ::PerspectiveSensorBase(const Spectrum &response,
                            const float aspect,
                            const Point &origin)
    :mAspectRatio(aspect),
     mInverseWindowSurfaceArea(1.0f/(2.0f*2.0f*aspect)),
     mWindowOrigin(origin),
     mResponse(response)
{
  ;
} // end PerspectiveSensorBase::PerspectiveSensorBase()

template<typename V3, typename S3>
  void PerspectiveSensorBase<V3,S3>
    ::set(const float aspect,
          const Point &origin)
{
  mAspectRatio = aspect;
  mWindowOrigin = origin;
  mInverseWindowSurfaceArea = (1.0f/(2.0f*2.0f*mAspectRatio));
} // end PerspectiveSensorBase::set()

template<typename V3, typename S3>
  S3 PerspectiveSensorBase<V3,S3>
    ::sample(const Point &point,
             const Point &tangent,
             const Point &binormal,
             const Point &normal,
             const float u0,
             const float u1,
             const float u2,
             Vector &ws,
             float &pdf,
             bool &delta,
             unsigned int &component) const
{
  component = 0;
  return sample(point,tangent,binormal,normal,u0,u1,u2,ws,pdf,delta);
} // end PerspectiveSensorBase::sample()

template<typename V3, typename S3>
  S3 PerspectiveSensorBase<V3,S3>
    ::sample(const Point &point,
             const Point &tangent,
             const Point &binormal,
             const Point &normal,
             const float u0,
             const float u1,
             const float u2,
             Vector &ws,
             float &pdf,
             bool &delta) const
{
  delta = false;
  Point q;
  sampleWindow(u0,u1,
               binormal,
               tangent,
               normal,
               q,
               pdf);
  ws = q - point;
  float d2 = dot(ws,ws);
  ws /= sqrtf(d2);

  // compute surface area pdf to solid angle pdf
  pdf *= d2;

  // this removes the vignetting effect, but is it correct?
  pdf *= dot(normal,ws);

  //// divide the response by the dot product to remove the vignette effect
  //return mResponse / dot(dg.getNormal(),ws);
  return mResponse;
} // end PerspectiveSensorBase::sample()

template<typename V3, typename S3>
  void PerspectiveSensorBase<V3,S3>
    ::sampleWindow(const float u,
                   const float v,
                   const Vector &xAxis,
                   const Vector &yAxis,
                   const Vector &zAxis,
                   Point &p,
                   float &pdf) const
{
  p = mWindowOrigin;
  p += 2.0f * mAspectRatio * u * xAxis;
  p += 2.0f * v * yAxis;
  pdf = mInverseWindowSurfaceArea;
} // end PerspectiveSensorBase::sampleWindow()

template<typename V3, typename S3>
  float PerspectiveSensorBase<V3,S3>
    ::evaluatePdf(const Vector &ws,
                  const Point &point,
                  const Vector &tangent,
                  const Vector &binormal,
                  const Vector &normal) const
{
  // intersect a ray through point in direction ws with the sensor window
  // remember that the normal points in the -look direction
  float t = -dot(normal, mWindowOrigin - point) /
            -dot(normal, ws);

  // if t is negative, then ws came from 'behind the camera',
  // and there is zero pdf of generating such directions
  if(t < 0) return 0;

  // compute q the intersection with the ray and the window
  Point q = point + t * ws;
  Point coords = q - mWindowOrigin;
  coords *= 0.5f;

  float u = dot(coords,binormal) / mAspectRatio;
  float v = dot(coords,tangent);

  // if the ray does not pass through the window,
  // then there is zero probability of having generated it
  if(u < 0.0f || u >= 1.0f) return 0.0f;
  if(v < 0.0f || v >= 1.0f) return 0.0f;

  // compute solid angle pdf
  Vector wi = q - point;
  float pdf = dot(wi,wi);
  pdf *= mInverseWindowSurfaceArea;

  // this removes the vignetting effect, but is it correct?
  pdf *= fabs(dot(normal,ws));

  return pdf;
} // end PerspectiveSensorBase::evaluatePdf()

template<typename V3, typename S3>
  S3 PerspectiveSensorBase<V3,S3>
    ::evaluate(const Vector &ws,
               const Point &point,
               const Vector &tangent,
               const Vector &binormal,
               const Vector &normal) const
{
  // evaluate the pdf at ws, and if it is not zero, return mResponse
  // divide by dot product to remove the vignetting effect
  //return evaluatePdf(ws, dg) > 0 ? (mResponse / dg.getNormal().absDot(ws)) : Spectrum::black();
  
  Spectrum result;
  result.x = 0;
  result.y = 0;
  result.z = 0;

  if(evaluatePdf(ws,point,tangent,binormal,normal) > 0)
  {
    result = mResponse;
  } // end if

  return result;
} // end PerspectiveSensorBase::evaluate()

template<typename V3, typename S3>
  void PerspectiveSensorBase<V3,S3>
    ::invert(const Vector &w,
             const Point &point,
             const Vector &tangent,
             const Vector &binormal,
             const Vector &normal,
             float &u0,
             float &u1) const
{
  // a ray from the dg in direction w intersects the window at
  // time t
  // remember that the normal points in the -look direction
  float t =
    -dot(normal,mWindowOrigin - point) /
    -dot(normal,w);

  // compute q the intersection with the ray and the plane
  Point q = point + t * w;

  // this is the inverse operation of sampleFilmPlane():
  // get the film plane coordinates of q
  q -= mWindowOrigin;
  q *= 0.5f;

  u0 = dot(q,binormal) / mAspectRatio;
  u1 = dot(q,tangent);
} // end PerspectiveSensorBase::invert()

