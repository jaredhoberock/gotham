/*! \file AshikhminShirleyReflectionBase.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a base class for
 *         a scattering function implementing an
 *         anisotropic Phong lobe.
 */

#pragma once

template<typename V3, typename S3, typename Boolean = bool>
  class AshikhminShirleyReflectionBase
{
  public:
    typedef V3 Vector;
    typedef S3 Spectrum;

    /*! Constructor accepts a reflectance, index of refraction,
     *  and two exponents to create an anisotropic glossy Fresnel conductor.
     *  \param r The reflectance of this PhongReflectionBase.
     *  \param eta The index of refraction of the Fresnel conductor.
     *  \param uExponent The Ashikhmin-Shirley exponent in the u direction.
     *  \param vExponent The Ashikhmin-Shriley exponent in the v direction.
     */
    inline AshikhminShirleyReflectionBase(const Spectrum &r,
                                          const float eta,
                                          const float uExponent,
                                          const float vExponent);

    /*! Constructor accepts a reflectance, two indices of refraction,
     *  and two exponents to create an anisotropic glossy Fresnel dielectric.
     *  \param r The reflectance of this PhongReflection.
     *  \param etai The index of refraction of the medium surrounding the dielectric.
     *  \param etat The index of refraction of the Fresnel dielectric medium.
     *  \param uExponent The Ashikhmin-Shirley exponent in the u direction.
     *  \param vExponent The Ashikhmin-Shriley exponent in the v direction.
     */
    inline AshikhminShirleyReflectionBase(const Spectrum &r,
                                          const float etai,
                                          const float etat,
                                          const float uExponent,
                                          const float vExponent);

    /*! This method evaluates this AshikhminShirleyReflectionBase function.
     *  \param wo The scattering direction.
     *  \param tangent The tangent direction.
     *  \param binormal The binormal direction.
     *  \param normal The normal direction.
     *  \param wi The incoming direction.
     *  \return The scattering in direction wo.
     */
    inline Spectrum evaluate(const Vector &wo,
                             const Vector &tangent,
                             const Vector &binormal,
                             const Vector &normal,
                             const Vector &wi) const;

    /*! This method samples this AshikhminShirleyReflectionBase given a Wo,
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
     *  \param delta This is set to false; AshikhminShirleyReflectionBase is not a delta function.
     *  \param component This is set to 0; AshikhminShirleyReflectionBase has only 1 component.
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
                           Boolean &delta,
                           unsigned int &component) const;

    /*! This method samples this AshikhminShirleyReflectionBase given a Wo,
     *  differential geometry vectors, and three numbers in the unit interval.
     *  \param wo The direction of scattering.
     *  \param point Ignored.
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
                           const Vector &point,
                           const Vector &tangent,
                           const Vector &binormal,
                           const Vector &normal,
                           const float u0,
                           const float u1,
                           const float u2,
                           Vector &wi,
                           float &pdf,
                           Boolean &delta,
                           unsigned int &component) const;

    /*! This method evaluates the pdf of choosing direction wi
     *  given direction wo and differential geometry vectors at the sampling point.
     *  \param wo The direction of scattering.
     *  \param tangent The tangent direction.
     *  \param binormal The binormal direction.
     *  \param normal The normal direction.
     *  \param wi The direction whose pdf is of interest.
     *  \return The pdf of wi given wo, tangent, binormal, normal, and wi.
     */
    inline float evaluatePdf(const Vector &wo,
                             const Vector &tangent,
                             const Vector &binormal,
                             const Vector &normal,
                             const Vector &wi) const;

    /*! This method evaluates the pdf of choosing direction wi
     *  given direction wo and differential geometry vectors at the sampling point.
     *  \param wo The direction of scattering.
     *  \param point Ignored.
     *  \param tangent The tangent direction.
     *  \param binormal The binormal direction.
     *  \param normal The normal direction.
     *  \param wi The direction whose pdf is of interest.
     *  \return The pdf of wi given wo, tangent, binormal, normal, and wi.
     */
    inline float evaluatePdf(const Vector &wo,
                             const Vector &point,
                             const Vector &tangent,
                             const Vector &binormal,
                             const Vector &normal,
                             const Vector &wi) const;

  protected:
    Spectrum mReflectance;
    float mEtat, mEtai;
    float mNu, mNv;
    Spectrum mAbsorptionCoefficient;
}; // end AshikhminShirleyReflectionBase

#include "AshikhminShirleyReflectionBase.inl"

