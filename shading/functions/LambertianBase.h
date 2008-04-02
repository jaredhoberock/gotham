/*! \file LambertianBase.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to
 *         a simple class encapsulating
 *         the Lambertian BRDF.
 */

#pragma once

template<typename V3, typename S3, typename DG>
  class LambertianBase
{
  public:
    typedef V3 Vector;
    typedef S3 Spectrum;
    typedef DG DifferentialGeometry;

    /*! Constructor accepts an albedo.
     *  \param albedo Sets mAlbedo.
     */
    inline LambertianBase(const Spectrum &albedo);

    /*! This method computes Lambertian scattering.
     *  \param wo The scattering direction.
     *  \param dg The DifferentialGeometry at the Point to be shaded.
     *  \param wi The incoming direction.
     *  \return mAlbedo / PI if wi & wo are in the same hemisphere;
     *          0, otherwise.
     */
    inline Spectrum evaluate(const Vector &wo,
                             const DifferentialGeometry &dg,
                             const Vector &wi) const;

    /*! This method returns a const reference to mAlbedo.
     *  \return mAlbedo.
     */
    inline const Spectrum &getAlbedo(void) const;

  protected:
    Spectrum mAlbedo;
    Spectrum mAlbedoOverPi;
}; // end LambertianBase

#include "LambertianBase.inl"

