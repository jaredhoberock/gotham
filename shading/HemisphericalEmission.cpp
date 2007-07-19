/*! \file HemisphericalEmission.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of HemisphericalEmission class.
 */

#include "HemisphericalEmission.h"

HemisphericalEmission
  ::HemisphericalEmission(const Spectrum &radiosity)
    :mRadiance(radiosity / PI)
{
  ;
} // end HemisphericalEmission::HemisphericalEmission()

Spectrum HemisphericalEmission
  ::evaluate(const Vector3 &w,
             const DifferentialGeometry &dg) const
{
  return w.dot(dg.getNormal()) > 0 ? mRadiance : Spectrum::black();
} // end HemisphericalEmission::evaluate()

