/*! \file areSameHemisphere.inl
 *  \author Jared Hoberock
 *  \brief Inline file for areSameHemisphere.h.
 */

#include "areSameHemisphere.h"

template<typename V3, typename N3>
  bool areSameHemisphere(const V3 &wo,
                         const N3 &n,
                         const V3 &wi)
{
  return areSameHemisphere(dot(wo,n), dot(wi,n));
} // end areSameHemisphere()

bool areSameHemisphere(const float coso,
                       const float cosi)
{
  return (coso > 0) == (cosi > 0); 
} // end areSameHemisphere()

