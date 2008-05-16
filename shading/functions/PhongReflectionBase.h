/*! \file PhongReflectionBase.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a base class
 *         for a scattering function implementing
 *         a Phong lobe.
 */

#pragma once

template<typename V3, typename S3>
  class PhongReflectionBase
{
  public:
    typedef V3 Vector;
    typedef S3 Spectrum;

    /*! Constructor accepts a reflectance, index of refraction,
     *  and a Phong exponent to create a glossy Fresnel conductor.
     *  \param r The reflectance of this PhongReflectionBase.
     *  \param eta The index of refraction of the Fresnel conductor.
     *  \param exponent The Phong exponent of this PhongReflection.
     */
    inline PhongReflectionBase(const Spectrum &r,
                               const float eta,
                               const float exponent);

    /*! Constructor accepts a reflectance, two indices of refraction,
     *  and a Phong exponent to create a glossy Fresnel dielectric.
     *  \param r The reflectance of this PhongReflection.
     *  \param etai The index of refraction of the medium surrounding the dielectric.
     *  \param etat The index of refraction of the Fresnel dielectric medium.
     *  \param exponent The Phong exponent of this PhongReflection.
     */
    inline PhongReflectionBase(const Spectrum &r,
                               const float etai,
                               const float etat,
                               const float exponent);

    /*! This method evaluates this PhongReflectionBase function.
     *  \param wo The scattering direction.
     *  \param normal The normal direction.
     *  \param wi The incoming direction.
     *  \return The scattering in direction wo.
     */
    inline Spectrum evaluate(const Vector &wo,
                             const Vector &normal,
                             const Vector &wi) const;

    /*! This method samples this PhongReflectionBase given a Wo,
     *  differential geometry vectors, and three numbers in the unit interval.
     *  \param wo The direction of scattering.
     *  \param tangent The tangent direction.
     *  \param binormal The binormal direction.
     *  \param normal The normal direction.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param wi The direction of scattering is returned here.
     *  \param pdf The value of the pdf at (u0,u1,u2) is returned here.
     *  \param delta This is set to false; PhongReflectionBase is not a delta function.
     *  \param component This is set to 0; PhongReflectionBase has only 1 component.
     */
    inline Spectrum sample(const Vector &wo,
                           const Vector &tangent,
                           const Vector &binormal,
                           const Vector &normal,
                           const float u0,
                           const float u1,
                           const float u2,
                           Vector &wi,
                           float &pdf,
                           bool &delta,
                           unsigned int &component) const;

  protected:
    Spectrum mReflectance;
    float mEtat, mEtai;
    float mExponent;
    Spectrum mAbsorptionCoefficient;
}; // end PhongReflectionBase

#include "PhongReflectionBase.inl"

