/*! \file SpecularReflectionBase.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a scattering function
 *         implementing specular reflection.
 */

#pragma once

template<typename V3, typename S3, typename Boolean = bool>
  class SpecularReflectionBase
{
  public:
    typedef V3 Vector;
    typedef S3 Spectrum;

    /*! This constructor creates a SpecularReflectionBase conductor.
     *  \param r Sets mReflectance.
     *  \param eta Sets the index of refraction of the Fresnel conductor.
     */
    inline SpecularReflectionBase(const Spectrum &t,
                                  const float eta);

    /*! This constructor creates a SpecularReflectionBase dielectric.
     *  \param r Sets mReflectance.
     *  \param etai Sets the index of refraction of the space surrounding the dielectric.
     *  \param etat Sets the index of refraction of the Fresnel dielectric medium.
     */
    inline SpecularReflectionBase(const Spectrum &r, const float etai, const float etat);

    /*! This method evaluates this SpecularReflectionBase function.
     *  \return Black; the response is non-zero only in
     *          the specular direction.
     */
    inline Spectrum evaluate(void) const;

    /*! This method is included to match the uniform evaluate() interface.
     *  \param wo Ignored.
     *  \param normal Ignored.
     *  \param wi Ignored.
     *  \return evaluate()
     */
    inline Spectrum evaluate(const Vector &wo,
                             const Vector &normal,
                             const Vector &wi) const;

    /*! This method samples this SpecularTransmission function given a wo,
     *  DifferentialGeometry, and three numbers in the unit interval.
     *  \param wo The direction of scattering.
     *  \param normal The normal direction of the surface at
     *         the point of interest.
     *  \param wi The direction of scattering is returned here.
     *  \param pdf The value of the pdf at wi is returned here.
     *  \param delta This is set to true.
     *  \param component This is set to 0.
     *  \return The bidirectional scattering from wi to wo is returned here.
     */
    inline Spectrum sample(const Vector &wo,
                           const Vector &normal,
                           Vector &wi,
                           float &pdf,
                           Boolean &delta,
                           unsigned int &component) const;

    /*! This method is included to conform to a uniform interface for
     *  scattering functions. Many of the parameters are ignored.
     *  This method samples this SpecularTransmission function given a wo,
     *  DifferentialGeometry, and three numbers in the unit interval.
     *  \param wo The direction of scattering.
     *  \param tangent Ignored.
     *  \param binormal Ignored.
     *  \param normal The normal direction of the surface at
     *         the point of interest.
     *  \param u0 Ignored.
     *  \param u1 Ignored.
     *  \param u2 Ignored.
     *  \param wi The direction of scattering is returned here.
     *  \param pdf The value of the pdf at wi is returned here.
     *  \param delta This is set to true.
     *  \param component This is set to 0.
     *  \return The bidirectional scattering from wi to wo is returned here.
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

    /*! This method returns 0.0f.
     *  \param wo A Vector pointing towards the direction of scattering.
     *  \param point Ignored.
     *  \param tangent Ignored.
     *  \param binormal Ignored.
     *  \param normal Ignored.
     *  \param wi A Vector pointing towards the direction of incidence.
     *  \return 0.0f.
     */
    inline float evaluatePdf(const Vector &wo,
                             const Vector &point,
                             const Vector &tangent,
                             const Vector &binormal,
                             const Vector &normal,
                             const Vector &wi) const;

  protected:
    Spectrum mReflectance;
    float mEtai, mEtat;
    Spectrum mAbsorptionCoefficient;
}; // end SpecularReflectionBase

#include "SpecularReflectionBase.inl"

