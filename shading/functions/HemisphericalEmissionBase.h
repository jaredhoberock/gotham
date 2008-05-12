/*! \file HemisphericalEmissionBase.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a
 *         simple class encapsulating
 *         the hemispherical emission
 *         function.
 */

#pragma once

template<typename V3, typename S3, typename DG>
  class HemisphericalEmissionBase
{
  public:
    typedef V3 Vector;
    typedef S3 Spectrum;
    typedef DG DifferentialGeometry;

    /*! Constructor accepts a radiosity.
     *  \param radiosity The sum of radiance emitted from this
     *         hemisphere. mRadiance is set to radiosity / PI
     */
    inline HemisphericalEmissionBase(const Spectrum &radiosity);

    /*! This method computes hemispherical emission.
     *  \param w  The outgoing direction of emission.
     *  \param dg The DifferentialGeometry at the Point of interest.
     *  \return mRadiance if w points in the direction of dg's normal direction.
     */
    inline Spectrum evaluate(const Vector &w,
                             const DifferentialGeometry &dg) const;

    /*! This method computes hemispherical emission.
     *  \param w  The outgoing direction of emission.
     *  \param normal The normal direction at the point of interest.
     *  \return mRadiance if w points in the direction of dg's normal direction.
     */
    inline Spectrum evaluate(const Vector &w,
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

