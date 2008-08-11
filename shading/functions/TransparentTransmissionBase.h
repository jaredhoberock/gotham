/*! \file TransparentTransmissionBase.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a function that implements
 *         a "pass-through" transmission function.
 */

#pragma once

template<typename V3, typename S3, typename Boolean = bool>
  class TransparentTransmissionBase
{
  public:
    typedef V3 Vector;
    typedef S3 Spectrum;

    /*! Constructor accepts a transmittance.
     *  \param transmittance Sets mTransmittance.
     */
    inline TransparentTransmissionBase(const Spectrum &transmittance);

    /*! This method evaluates this SpecularReflectionBase function.
     *  \return Black; the response is non-zero only in
     *          the pass-through direction.
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

    /*! This method samples this SpecularTransmission function given a wo.
     *  \param wo The direction of scattering.
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
    Spectrum mTransmittance;
}; // end TransparentTransmissionBase

#include "TransparentTransmissionBase.inl"

