/*! \file TransparentTransmission.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of TransparentTransmission class.
 */

#include "TransparentTransmission.h"

TransparentTransmission
  ::TransparentTransmission(const Spectrum &transmittance)
    :Parent0(),
     Parent1(transmittance)
{
  ;
} // end TransparentTransmission::TransparentTransmission()

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
  return Parent1::sample(wo,
                         dg.getPoint(),
                         dg.getTangent(),
                         dg.getBinormal(),
                         dg.getNormal(),
                         u0, u1, u2,
                         wi, pdf, delta, component);
} // end TransparentTransmission::sample()

Spectrum TransparentTransmission
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi) const
{
  return Parent1::evaluate(wo,
                           dg.getNormal(),
                           wi);
} // end TransparentTransmission::evaluate()

