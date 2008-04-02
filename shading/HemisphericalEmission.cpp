/*! \file HemisphericalEmission.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of HemisphericalEmission class.
 */

#include "HemisphericalEmission.h"

HemisphericalEmission
  ::HemisphericalEmission(const Spectrum &radiosity)
    :Parent0(),Parent1(radiosity)
{
  ;
} // end HemisphericalEmission::HemisphericalEmission()

Spectrum HemisphericalEmission
  ::evaluate(const Vector3 &w,
             const DifferentialGeometry &dg) const
{
  return Parent1::evaluate(w,dg);
} // end HemisphericalEmission::evaluate()

