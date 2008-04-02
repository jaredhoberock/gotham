/*! \file areSameHemisphere.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to some functions
 *         for determining whether or not two vectors
 *         lie in the same hemisphere with respect to
 *         a reference.
 */

#pragma once

/*! This function evaluates whether or not an incident and exitant direction
 *  are in the same hemisphere with respect to the Normal direction.
 *  \param wi A vector pointing towards the direction of incident radiance.
 *  \param n The surface Normal at the point of interest.
 *  \param wo A vector pointing towards the direction of exitant radiance.
 *  \return true if wi and wo are in the same hemisphere with respect to
 *          n; false, otherwise.
 */
template<typename V3, typename N3>
  inline bool areSameHemisphere(const V3 &wo,
                                const N3 &n,
                                const V3 &wi);

inline float areSameHemisphere(const float coso,
                               const float cosi);

#include "areSameHemisphere.inl"

