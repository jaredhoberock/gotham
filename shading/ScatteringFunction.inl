/*! \file ScatteringFunction.inl
 *  \author Jared Hoberock
 *  \brief Inline file for ScatteringFunction.h.
 */

#include "ScatteringFunction.h"

bool ScatteringFunction
  ::areSameHemisphere(const Vector3 &wo,
                      const Vector3 &wi)
{
  return wi[2] * wo[2] > 0;
} // end ScatteringFunction::areSameHemisphere()

