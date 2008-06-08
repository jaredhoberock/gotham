/*! \file PerspectiveSensor.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PerspectiveSensor class.
 */

#include "PerspectiveSensor.h"

PerspectiveSensor
  ::PerspectiveSensor(void)
    :Parent0(),Parent1()
{
  ;
} // end PerspectiveSensor::PerspectiveSensor()

PerspectiveSensor
  ::PerspectiveSensor(const Spectrum &response,
                      const float aspect,
                      const Point &origin)
    :Parent0(),Parent1(response,aspect,origin)
{
  ;
} // end PerspectiveSensor::PerspectiveSensor()

Spectrum PerspectiveSensor
  ::sample(const DifferentialGeometry &dg,
           const float u0,
           const float u1,
           const float u2,
           Vector &ws,
           float &pdf,
           bool &delta) const
{
  return Parent1::sample(dg.getPoint(),dg.getTangent(),dg.getBinormal(),dg.getNormal(),u0,u1,u2,ws,pdf,delta);
} // end PerspectiveSensor::sample()

void PerspectiveSensor
  ::invert(const Vector &w,
           const DifferentialGeometry &dg,
           float &u0,
           float &u1) const
{
  return Parent1::invert(w,dg.getPoint(),dg.getTangent(),dg.getBinormal(),dg.getNormal(),u0,u1);
} // end PerspectiveSensor::invert()

float PerspectiveSensor
  ::evaluatePdf(const Vector &ws,
                const DifferentialGeometry &dg) const
{
  return Parent1::evaluatePdf(ws,dg.getPoint(),dg.getTangent(),dg.getBinormal(),dg.getNormal());
} // end PerspectiveSensor::evaluatePdf()

Spectrum PerspectiveSensor
  ::evaluate(const Vector &ws,
             const DifferentialGeometry &dg) const
{
  return Parent1::evaluate(ws,dg.getPoint(),dg.getTangent(),dg.getBinormal(),dg.getNormal());
} // end PerspectiveSensor::evaluate()

