/*! \file Mappings.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Mappings class.
 */

#include "Mappings.h"
#include <2dmapping/UnitSquareToHemisphere.h>
#include <2dmapping/UnitSquareToCosineHemisphere.h>
#include <2dmapping/UnitSquareToPhongHemisphere.h>
#include <2dmapping/UnitSquareToAnisotropicLobe.h>
#include <assert.h>
#include "Transform.h"

void Mappings
  ::unitSquareToHemisphere(const float u0,
                           const float u1,
                           const Vector3 &xAxis,
                           const Vector3 &yAxis,
                           const Vector3 &zAxis,
                           Vector3 &w,
                           float &pdf)
{
  // sample a point on the canonical +z hemisphere
  Vector3 temp;
  UnitSquareToHemisphere::evaluate(u0,u1,temp,&pdf);

  // M = x[0] x[1] x[2]
  //     y[0] y[1] y[2]
  //     z[0] z[1] z[2]

  // w = M^T * temp

  // rotate into our local frame
  w[0] = temp.dot(Vector3(xAxis[0], yAxis[0], zAxis[0]));
  w[1] = temp.dot(Vector3(xAxis[1], yAxis[1], zAxis[1]));
  w[2] = temp.dot(Vector3(xAxis[2], yAxis[2], zAxis[2]));
} // end Mappings::unitSquareToHemisphere()

void Mappings
  ::unitSquareToCosineHemisphere(const float u0,
                                 const float u1,
                                 const Vector3 &xAxis,
                                 const Vector3 &yAxis,
                                 const Vector3 &zAxis,
                                 Vector3 &w,
                                 float &pdf)
{
  // sample a point on the canonical +z hemisphere
  Vector3 temp;
  UnitSquareToCosineHemisphere::evaluate(u0,u1,temp,&pdf);

  // M = x[0] x[1] x[2]
  //     y[0] y[1] y[2]
  //     z[0] z[1] z[2]

  // w = M^T * temp

  // rotate into our local frame
  w[0] = temp.dot(Vector3(xAxis[0], yAxis[0], zAxis[0]));
  w[1] = temp.dot(Vector3(xAxis[1], yAxis[1], zAxis[1]));
  w[2] = temp.dot(Vector3(xAxis[2], yAxis[2], zAxis[2]));
} // end Mappings::unitSquareToHemisphere()

void Mappings
  ::cosineHemisphereToUnitSquare(const Vector &w,
                                 const Vector &xAxis,
                                 const Vector &yAxis,
                                 const Vector &zAxis,
                                 float &u0,
                                 float &u1)
{
  // rotate w from its frame into a canonical one aligned with +z
  Vector temp;
  temp[0] = w.dot(xAxis);
  temp[1] = w.dot(yAxis);
  temp[2] = w.dot(zAxis);

  UnitSquareToCosineHemisphere::inverse(temp,u0,u1);
} // end Mappings::cosineHemisphereToUnitSquare()

float Mappings
  ::evaluateCosineHemispherePdf(const Vector &w,
                                const Vector &zAxis)
{
  return std::max(0.0f, w.dot(zAxis) * INV_PI);
} // end Mappings::evaluateCosineHemispherePdf()

void Mappings
  ::unitSquareToPhongLobe(const float u0,
                          const float u1,
                          const Vector &r,
                          const float exponent,
                          const Vector &xAxis,
                          const Vector &yAxis,
                          const Vector &zAxis,
                          Vector &w,
                          float &pdf)
{
  // XXX fix this
  // generate a random vector on the +z hemisphere
  UnitSquareToPhongHemisphere::evaluate<float, Vector>(u0,u1,exponent,w,&pdf);

  // get the angle between r and (0,0,1)
  // r dot (0,0,1) = r[0]*0 + r[1]*0 + r[2]*1
  float angle = acosf(r[2]);

  if(angle > 0.0001 && angle < PI - 0.0001f)
  {
    // get the vector orthogonal to r and (0,0,1)
    Vector axis = Vector(0,0,1).cross(r);

    // rotate result about axis
    w = Transform::rotateVector(DEGREES(angle),
                                axis[0], axis[1], axis[2],
                                w);
  } // end if
  else if(r[2] < 0)
  {
    // no rotation necessary, just flip because r == -z
    w = -w;
  } // end else if
} // end Mappings::unitSquareToPhongLobe()

float Mappings
  ::evaluatePhongLobePdf(const Vector &w,
                         const Vector &r,
                         const float exponent)
{
  // from Walter et al, 2007, equation 30
  return (exponent + 2.0f) * powf(w.dot(r), exponent) * INV_TWOPI;
} // end Mappings::evaluatePhongLobePdf()

void Mappings
  ::unitSquareToAnisotropicPhongLobe(const float u0,
                                     const float u1,
                                     const float nu,
                                     const float nv,
                                     const Vector &xAxis,
                                     const Vector &yAxis,
                                     const Vector &zAxis,
                                     Vector &w,
                                     float &pdf)
{
  // generate a random vector on the +z hemisphere
  Vector temp;
  UnitSquareToAnisotropicLobe::evaluate<float, Vector>(u0,u1,nu,nv,temp,&pdf);

  // M = x[0] x[1] x[2]
  //     y[0] y[1] y[2]
  //     z[0] z[1] z[2]

  // w = M^T * temp

  // rotate into our local frame
  w[0] = temp.dot(Vector3(xAxis[0], yAxis[0], zAxis[0]));
  w[1] = temp.dot(Vector3(xAxis[1], yAxis[1], zAxis[1]));
  w[2] = temp.dot(Vector3(xAxis[2], yAxis[2], zAxis[2]));
} // end Mappings::unitSquareToPhongLobe()

float Mappings
  ::evaluateAnisotropicPhongLobePdf(const Vector &w,
                                    const float nu,
                                    const float nv,
                                    const Vector &xAxis,
                                    const Vector &yAxis,
                                    const Vector &zAxis)
{
  float costheta = zAxis.absDot(w);
  float x = xAxis.absDot(w);
  float y = yAxis.absDot(w);
  float e = (nu * x * x + nv * y * y) / (1.0f - costheta * costheta);
  return sqrtf((nu+2.0f)*(nv+2.0f)) * INV_TWOPI * powf(costheta,e);
} // end Mappings::evaluateAnisotropicPhongLobePdf()

