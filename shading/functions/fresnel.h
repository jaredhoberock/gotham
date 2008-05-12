/*! \file fresnel.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a set
 *         of functions which evaluate
 *         the Fresnel effect.
 */

#pragma once

/*! This function evaluates a Fresnel dielectric function.
 *  \param ei The incident index of refraction.
 *  \param et The transmitted index of refraction.
 *  \param cosi The cosine of the angle between the incident
 *              direction and normal direction.
 *  \param cost The cosine of the angle between the transmitted
 *              direction and normal direction.
 */
inline float evaluateFresnelDielectric(const float ei,
                                       const float et,
                                       const float cosi,
                                       const float cost);

#include "fresnel.inl"

