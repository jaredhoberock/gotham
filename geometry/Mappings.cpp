/*! \file Mappings.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Mappings class.
 */

#include "Mappings.h"
#include <2dmapping/UnitSquareToHemisphere.h>
#include <2dmapping/UnitSquareToCosineHemisphere.h>

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

