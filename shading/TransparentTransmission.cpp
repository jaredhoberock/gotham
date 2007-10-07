/*! \file TransparentTransmission.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of TransparentTransmission class.
 */

#include "TransparentTransmission.h"

Spectrum TransparentTransmission
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi) const
{
  return Spectrum::black();
} // end TransparentTransmission::evaluate()

Spectrum TransparentTransmission
  ::sample(const Vector &wo,
           const DifferentialGeometry &dg,
           const float u0,
           const float u1,
           const float u2,
           Vector &wi,
           float &pdf,
           bool &delta,
           ComponentIndex &component) const
{
  delta = true;
  component = 0;

  wi = -wo;
  pdf = 1.0f;

  return mTransmittance / dg.getNormal().absDot(wi);
} // end TransparentTransmission::sample()

