/*! \file HemisphericalEmissionBase.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a
 *         simple class encapsulating
 *         the hemispherical emission
 *         function.
 */

#pragma once

template<typename V3, typename S3>
  class HemisphericalEmissionBase
{
  public:
    typedef V3 Vector;
    typedef S3 Spectrum;

    /*! Constructor accepts a radiosity.
     *  \param radiosity The sum of radiance emitted from this
     *         hemisphere. mRadiance is set to radiosity / PI
     */
    inline HemisphericalEmissionBase(const Spectrum &radiosity);

    /*! This method computes hemispherical emission.
     *  \param w  The outgoing direction of emission.
     *  \param normal The normal direction at the point of interest.
     *  \return mRadiance if w points in the direction of dg's normal direction.
     */
    inline Spectrum evaluate(const Vector &w,
                             const Vector &normal) const;

    /*! This method is included to conform to the standard scattering function
     *  interface.
     *  \param wo The outgoing direction of emission.
     *  \param point Ignored.
     *  \param tangent Ignored.
     *  \param binormal Ignored.
     *  \param normal Ignored.
     *  \return evaluate(wo,normal)
     */
    inline Spectrum evaluate(const Vector &wo,
                             const Vector &point,
                             const Vector &tangent,
                             const Vector &binormal,
                             const Vector &normal) const;

    /*! This method returns a const reference to mRadiance.
     *  \return mRadiance
     */
    inline const Spectrum &getRadiance(void) const;

    /*! This method returns PI * mRadiance.
     *  \return PI * mRadiance.
     */
    inline const Spectrum getRadiosity(void) const;

  protected:
    Spectrum mRadiance;
}; // end HemisphericalEmissionBase

#include "HemisphericalEmissionBase.inl"

