/*! \file ThinGlass.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ThinGlass class.
 */

#include "ThinGlass.h"

ThinGlass
  ::ThinGlass(const Spectrum &r,
              const Spectrum &t,
              const float etai,
              const float etat)
    :Parent(r,t,etai,etat)
{
  ;
} // end ThinGlass::ThinGlass()

Spectrum ThinGlass
  ::sampleTransmittance(const Vector &wo,
                        const DifferentialGeometry &dg,
                        Vector &wi) const
{
  wi = -wo;
  return mTransmittance;
} // end ThinGlass::sampleTransmittance()

Spectrum ThinGlass
  ::evaluateTransmittance(const Vector &wo,
                          const DifferentialGeometry &dg) const
{
  return mTransmittance;
} // end ThinGlass::evaluateTransmittance()

