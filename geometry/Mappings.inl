/*! \file Mappings.inl
 *  \author Jared Hoberock
 *  \brief Implementation of Mappings class.
 */

#include "Mappings.h"
#include <2dmapping/UnitSquareToSphere.h>
#include <2dmapping/UnitSquareToHemisphere.h>
#include <2dmapping/UnitSquareToCosineHemisphere.h>
#include <2dmapping/UnitSquareToPhongHemisphere.h>
#include <2dmapping/UnitSquareToAnisotropicLobe.h>

template<typename V3>
  void Mappings<V3>
    ::alignVector(const Vector &xAxis,
                  const Vector &yAxis,
                  const Vector &zAxis,
                  const Vector &w,
                  Vector &wPrime)
{
  // M = x[0] x[1] x[2]
  //     y[0] y[1] y[2]
  //     z[0] z[1] z[2]

  // wPrime = M^T * w
  // assemble each row of M and dot into w
  Vector row;

  // rotate into our local frame
  row.x = xAxis.x;
  row.y = yAxis.x;
  row.z = zAxis.x;
  wPrime.x = dot(w, row);

  row.x = xAxis.y;
  row.y = yAxis.y;
  row.z = zAxis.y;
  wPrime.y = dot(w, row);

  row.x = xAxis.z;
  row.y = yAxis.z;
  row.z = zAxis.z;
  wPrime.z = dot(w, row);
} // end Mappings::alignVector()

template<typename V3>
  void Mappings<V3>
    ::unitSquareToSphere(const float u0,
                         const float u1,
                         const Vector &xAxis,
                         const Vector &yAxis,
                         const Vector &zAxis,
                         Vector &w,
                         float &pdf)
{
  // sample a point on the canonical sphere
  Vector temp;
  UnitSquareToSphere::evaluate(u0,u1,temp,&pdf);

  // align with frenet frame
  alignVector(xAxis, yAxis, zAxis, temp, w);
} // end Mappings::unitSquareToHemisphere()

template<typename V3>
  void Mappings<V3>
    ::unitSquareToHemisphere(const float u0,
                             const float u1,
                             const Vector &xAxis,
                             const Vector &yAxis,
                             const Vector &zAxis,
                             Vector &w,
                             float &pdf)
{
  // sample a point on the canonical +z hemisphere
  Vector temp;
  UnitSquareToHemisphere::evaluate(u0,u1,temp,&pdf);

  // align with frenet frame
  alignVector(xAxis, yAxis, zAxis, temp, w);
} // end Mappings::unitSquareToHemisphere()

template<typename V3>
  void Mappings<V3>
    ::unitSquareToCosineHemisphere(const float u0,
                                   const float u1,
                                   const Vector &xAxis,
                                   const Vector &yAxis,
                                   const Vector &zAxis,
                                   Vector &w,
                                   float &pdf)
{
  // sample a point on the canonical +z hemisphere
  Vector temp;
  //UnitSquareToCosineHemisphere::evaluate(u0,u1,temp.x,temp.y,temp.z,pdf);
  UnitSquareToCosineHemisphere<float,V3>::evaluate(u0,u1,temp.x,temp.y,temp.z,pdf);

  // align with frenet frame
  alignVector(xAxis, yAxis, zAxis, temp, w);
} // end Mappings::unitSquareToHemisphere()

template<typename V3>
  void Mappings<V3>
    ::cosineHemisphereToUnitSquare(const Vector &w,
                                   const Vector &xAxis,
                                   const Vector &yAxis,
                                   const Vector &zAxis,
                                   float &u0,
                                   float &u1)
{
  // rotate w from its frame into a canonical one aligned with +z
  Vector temp;
  temp.x = dot(w,xAxis);
  temp.y = dot(w,yAxis);
  temp.z = dot(w,zAxis);

  UnitSquareToCosineHemisphere<float,Vector>::inverse(temp,u0,u1);
} // end Mappings::cosineHemisphereToUnitSquare()

template<typename V3>
  float Mappings<V3>
    ::evaluateCosineHemispherePdf(const Vector &w,
                                  const Vector &zAxis)
{
  float result = dot(w,zAxis) * INV_PI;
  result = result < 0 ? 0.0f : result;
  return result;
} // end Mappings::evaluateCosineHemispherePdf()

template<typename V3>
  void Mappings<V3>
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
  UnitSquareToPhongHemisphere<float,Vector>::evaluate(u0,u1,exponent,w,&pdf);

  // get the angle between r and (0,0,1)
  // r dot (0,0,1) = r[0]*0 + r[1]*0 + r[2]*1
  float angle = acosf(r.z);

  if(angle > 0.0001 && angle < PI - 0.0001f)
  {
    Vector temp = w;

    // get the vector orthogonal to r and (0,0,1)
    Vector zee;
    zee.x = 0;
    zee.y = 0;
    zee.z = 1;

    Vector axis = cross(zee,r);

    //// rotate result about axis
    //w = Transform::rotateVector(DEGREES(angle),
    //                            axis.x, axis.y, axis.z,
    //                            w);

    // axis should be unit length already
    Vector axis2 = axis * axis;
    float sine = sinf(-angle);
    float cosine = cos(-angle);
    float omcos = 1.0f - cosine;

    // assemble the rows of a rotation matrix
    Vector row;

    // first row
    row.x = axis2.x + (1.0f - axis2.x) * cosine;
    row.y = axis.x * axis.y * omcos + axis.z * sine;
    row.z = axis.x * axis.z * omcos - axis.y * sine;
    w.x = dot(row, temp);

    // second row
    row.x = axis.x * axis.y * omcos - axis.z * sine;
    row.y = axis2.y + (1.0f - axis2.y) * cosine;
    row.z = axis.y * axis.z * omcos + axis.x * sine;
    w.y = dot(row, temp);

    // third row
    row.x = axis.x * axis.z * omcos + axis.y * sine;
    row.y = axis.y * axis.z * omcos - axis.x * sine;
    row.z = axis2.z + (1.0f - axis2.z) * cosine;
    w.z = dot(row, temp);
  } // end if
  else if(r.z < 0)
  {
    // no rotation necessary, just flip because r == -z
    w.x = -w.x;
    w.y = -w.y;
    w.z = -w.z;
  } // end else if
} // end Mappings::unitSquareToPhongLobe()

template<typename V3>
  float Mappings<V3>
    ::evaluatePhongLobePdf(const Vector &w,
                           const Vector &r,
                           const float exponent)
{
  // from Walter et al, 2007, equation 30
  return (exponent + 2.0f) * powf(w.dot(r), exponent) * INV_TWOPI;
} // end Mappings::evaluatePhongLobePdf()

template<typename V3>
  void Mappings<V3>
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

  alignVector(xAxis, yAxis, xAxis, temp, w);
} // end Mappings::unitSquareToPhongLobe()

template<typename V3>
  float Mappings<V3>
    ::evaluateAnisotropicPhongLobePdf(const Vector &w,
                                      const float nu,
                                      const float nv,
                                      const Vector &xAxis,
                                      const Vector &yAxis,
                                      const Vector &zAxis)
{
  float costheta = fabs(dot(zAxis,w));
  float x = fabs(dot(xAxis,w));
  float y = fabs(dot(yAxis,w));
  float e = (nu * x * x + nv * y * y) / (1.0f - costheta * costheta);
  return sqrtf((nu+2.0f)*(nv+2.0f)) * INV_TWOPI * powf(costheta,e);
} // end Mappings::evaluateAnisotropicPhongLobePdf()

