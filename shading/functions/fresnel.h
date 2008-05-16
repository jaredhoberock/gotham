/*! \file fresnel.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a set
 *         of functions which evaluate
 *         the Fresnel effect.
 */

#pragma once

/*! This function approximates the Fresnel conductor absorption coefficient
 *  assuming index of refraction = 1.
 *  \param r The Fresnel reflectance at normal incidence.
 *  \return The approximate Fresnel absorption coefficient.
 */
template<typename S3>
  inline S3 approximateFresnelAbsorptionCoefficient(const S3 &r);

/*! This function evaluates a Fresnel dielectric function when cost
 *  is unknown.
 *  \param ei The incident index of refraction.
 *  \param et The transmitted index of refraction.
 *  \param cosi The cosine of the angle between the incident
 *              direction and normal direction.
 */
inline float evaluateFresnelDielectric(const float ei,
                                       const float et,
                                       const float cosi);

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

/*! This function evaluates a Fresnel dielectric function when the orientation
 *  of the mediums are unknown
 *  \param ei The index of refraction of the medium surrounding the dielectric.
 *  \param et The index of refraction of the dielectric medium.
 *  \param cosi The cosine of the angle between the incident
 *              direction and normal direction.
 */
inline float evaluateFresnelDielectricUnknownOrientation(float ei,
                                                         float et,
                                                         const float cosi);

/*! This function evaluates a Fresnel conductor function.
 *  \param absorption The approximate absorption coefficient of the
 *         Fresnel conductor.
 *  \param eta The index of refraction of the conductor.
 *  \param cosi The cosine of the angle between the incident
 *              direction and normal direction.
 */
template<typename S3>
  inline S3 evaluateFresnelConductor(const S3 &absorption,
                                     const float eta,
                                     const float cosi);

#include "fresnel.inl"

