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

    /*! This method samples this LambertianBase given a Wo,
     *  DifferentialGeometry, and three numbers in the unit interval.
     *  \param wo The direction of scattering.
     *  \param dg The DifferentialGeometry at the point of interest.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param wi The direction of scattering is returned here.
     *  \param pdf The value of the pdf at (u0,u1,u2) is returned here.
     *  \param delta This is set to false; LambertianBase is not a delta function.
     *  \param component This is set to 0; LambertianBase has only 1 component.
     */
    inline Spectrum sample(const Vector &wo,
                           const DifferentialGeometry &dg,
                           const float u0,
                           const float u1,
                           const float u2,
                           Vector &wi, float &pdf, bool &delta, unsigned int &component) const;

  protected:
    Spectrum mAlbedo;
    Spectrum mAlbedoOverPi;
}; // end LambertianBase

#include "LambertianBase.inl"

