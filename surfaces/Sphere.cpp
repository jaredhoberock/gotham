/*! \file Sphere.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Sphere class.
 */

#include "Sphere.h"
#include <quadratic/Quadratic.h>
#include "../geometry/BoundingBox.h"
#include "../geometry/Vector.h"
#include "../geometry/Ray.h"
#include "../geometry/DifferentialGeometry.h"
#include <2dmapping/UnitSquareToSphere.h>
#include <assert.h>

#ifndef PI
#define PI 3.14159265f
#endif // PI

#ifndef TWO_PI
#define TWO_PI 6.28318531f
#endif // TWO_PI

Sphere
  ::Sphere(void)
    :Parent()
{
  ;
} // end Sphere::Sphere()

Sphere
  ::Sphere(const Point &c, const float r)
    :Parent()
{
  mRadius = r;
  mCenter = c;
  mSurfaceArea = 4.0f * PI * mRadius*mRadius;
  mInverseSurfaceArea = 1.0f / mSurfaceArea;
} // end Sphere::Sphere()

float Sphere
  ::getSurfaceArea(void) const
{
  return mSurfaceArea;
} // end Sphere::getSurfaceArea()

float Sphere
  ::getInverseSurfaceArea(void) const
{
  return mInverseSurfaceArea;
} // end Sphere::getInverseSurfaceArea()

void Sphere
  ::getBoundingBox(BoundingBox &b) const
{
  b.set(mCenter - Point(mRadius, mRadius, mRadius),
        mCenter + Point(mRadius, mRadius, mRadius));
} // end Sphere::getBoundingBox()

bool Sphere
  ::intersect(const Ray &r) const
{
  Vector diff = r.getAnchor() - mCenter;

  // compute the coefficients of the quadratic equation of this Sphere
  float a = r.getDirection().norm2();
  float b = 2.0f * r.getDirection().dot(diff);
  float c = diff.norm2() - mRadius*mRadius;

  // solve the quadratic
  float root0, root1;
  Quadratic q(a,b,c);
  if(q.realRoots(root0,root1) == 0)
  {
    return false;
  } // end if

  // the hits must line in the legal bound
  if(root0 > r.getInterval()[1] ||
     root1 < r.getInterval()[0])
  {
    return false;
  } // end if

  // if the first root is behind the Ray, we want the second root
  if(root0 < r.getInterval()[0])
  {
    if(root0 > r.getInterval()[1])
    {
      return false;
    } // end if
  } // end if

  return true;
} // end Sphere::intersect()

bool Sphere
  ::intersect(const Ray &r,
              float &t,
              DifferentialGeometry &dg) const
{
  Vector diff = r.getAnchor() - mCenter;

  // compute the coefficients of the quadratic equation of this Sphere
  float a = r.getDirection().norm2();
  float b = 2.0f * r.getDirection().dot(diff);
  float c = diff.norm2() - mRadius*mRadius;

  // solve the quadratic
  float root0, root1;
  Quadratic q(a,b,c);
  if(q.realRoots(root0,root1) == 0)
  {
    return false;
  } // end if

  // the hits must line in the legal bound
  if(root0 > r.getInterval()[1] ||
     root1 < r.getInterval()[0])
  {
    return false;
  } // end if

  // the hits must lie in the legal bound
  if(root0 < r.getInterval()[0])
  {
    t = root1;
    if(t > r.getInterval()[1])
    {
      return false;
    } // end if
  } // end if
  else
  {
    t = root0;
  } // end else

  // compute the hit location
  Point p = r(t);

  getDifferentialGeometry(p, dg);

  return true;
} // end Sphere::intersect()

void Sphere
  ::getDifferentialGeometry(const Point &p,
                            DifferentialGeometry &dg) const
{
  float minTheta = PI;
  float maxTheta = 0;

  dg.setPoint(p);

  Vector n = (p-mCenter).normalize();
  dg.setNormal(n);

  // compute the parametric location
  float phi = atan2f(n[1], n[0]);
  if(phi < 0) phi += TWO_PI;

  ParametricCoordinates &uv = dg.getParametricCoordinates();
  uv[0] = phi / TWO_PI;
  //float theta = acosf(n[2] / mRadius);
  float theta = acosf(n[2]);
  uv[1] = (theta - minTheta) / (maxTheta - minTheta);

  // compute partial derivatives
  // 1. get partial derivatives for the hit point
  float zRadius = sqrtf(n[0]*n[0] + n[1]*n[1]);
  float invZRadius, cosPhi, sinPhi;

  Vector dpdu, dpdv;

  if(zRadius == 0)
  {
    // handle singularity
    cosPhi = 0;
    sinPhi = 1;

    dpdv = (maxTheta - minTheta) * Vector(n[2] * cosPhi, n[2] * sinPhi, -mRadius * sinf(theta));
    dpdu = dpdv.cross(n);
  } // end if
  else
  {
    invZRadius = 1.0f / zRadius;
    cosPhi = p[0] * invZRadius;
    sinPhi = p[1] * invZRadius;

    // dpdu
    dpdu = Vector(-TWO_PI * n[1], TWO_PI * n[0], 0);
       
    // dpdv
    dpdv = Vector(n[2]*cosPhi, n[2]*sinPhi, -mRadius * sinf(minTheta + uv[1]*(maxTheta - minTheta)));
    dpdv *= (maxTheta - minTheta);
  } // end else

  dg.setPointPartials(dpdu,dpdv);

  // 2. get partial derivatives for the Normal
  Vector d2Pduu = -(TWO_PI * TWO_PI) * Vector(n[0], n[1], 0);
  Vector d2Pduv = (maxTheta - minTheta) * n[2] * TWO_PI * Vector(-sinPhi, cosPhi, 0.0);
  Vector d2Pdvv = -(maxTheta - minTheta) * (maxTheta - minTheta) * Vector(n[0], n[1], n[2]);
  // compute coefficients for fundamental forms
  float E = dpdu.dot(dpdu);
  float F = dpdu.dot(dpdv);
  float G = dpdv.dot(dpdv);
  Vector N = dpdu.cross(dpdv);
  float e = N.dot(d2Pduu);
  float f = N.dot(d2Pduv);
  float g = N.dot(d2Pdvv);

  // finally, compute Normal partials from fundamental form coefficients
  float invEGF2 = 1.0f / (E*G - F*F);
  Vector &dndu = dg.getNormalVectorPartials()[0];
  dndu = (f*F - e*G) * invEGF2 * dpdu +
         (e*F - f*E) * invEGF2 * dpdv;
  Vector &dndv = dg.getNormalVectorPartials()[1];
  dndv = (g*F - f*G) * invEGF2 * dpdu +
         (f*F - g*E) * invEGF2 * dpdv;

  dg.setTangent(dg.getDPDU().normalize());

  // force an orthonormal basis
  dg.setBinormal(dg.getNormal().cross(dg.getTangent()));

  // set the inverse surface area
  dg.setInverseSurfaceArea(getInverseSurfaceArea());

  assert(dg.getTangent()[0] == dg.getTangent()[0]);
  assert(dg.getBinormal()[0] == dg.getBinormal()[0]);
} // end Sphere::getDifferentialGeometry()

float Sphere
  ::evaluateSurfaceAreaPdf(const DifferentialGeometry &dg) const
{
  return getInverseSurfaceArea();
} // end Sphere::evaluateSurfaceAreaPdf()

void Sphere
  ::sampleSurfaceArea(const float u1, const float u2, const float u3,
                      DifferentialGeometry &dg,
                      float &pdf) const
{
  Point p;
  UnitSquareToSphere::evaluate(u1,u2,p);

  // scale by radius
  p *= mRadius;

  // bias by center
  p += mCenter;

  pdf = getInverseSurfaceArea();

  getDifferentialGeometry(p,dg);
} // end Sphere::sampleSurfaceArea()

