/*! \file NullScatteringBase.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a black
 *         scattering function.
 */

#pragma once

template<typename V3, typename S3, typename DG>
  class NullScatteringBase
{
  public:
    typedef V3 Vector;
    typedef S3 Spectrum;
    typedef DG DifferentialGeometry;

    /*! This method returns zero.
     */
    inline Spectrum evaluate(void) const;

    /*! This method samples this NullScatteringBase given
     *  a DifferentialGeometry, and three numbers in the unit interval.
     *  \param dg The DifferentialGeometry at the point of interest.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param wi The direction of scattering is returned here.
     *  \param pdf The value of the pdf at (u0,u1,u2) is returned here.
     *  \param delta This is set to false; LambertianBase is not a delta function.
     *  \param component This is set to 0; LambertianBase has only 1 component.
     */
    inline Spectrum sample(const DifferentialGeometry &dg,
                           const float u0,
                           const float u1,
                           const float u2,
                           Vector &wi,
                           float &pdf,
                           bool &delta,
                           unsigned int &component) const;

    /*! This method samples this NullScatteringBase given 
     *  differential geometry and three numbers in the unit interval.
     *  \param wo The direction of scattering.
     *  \param tangent The tangent direction.
     *  \param binormal The binormal direction.
     *  \param normal The normal direction.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param wi The direction of scattering is returned here.
     *  \param pdf The value of the pdf at (u0,u1,u2) is returned here.
     *  \param delta This is set to false; LambertianBase is not a delta function.
     *  \param component This is set to 0; LambertianBase has only 1 component.
     */
    inline Spectrum sample(const Vector &tangent,
                           const Vector &binormal,
                           const Vector &normal,
                           const float u0,
                           const float u1,
                           const float u2,
                           Vector &wi,
                           float &pdf,
                           bool &delta,
                           unsigned int &component) const;

    inline Spectrum sample(const Vector &tangent,
                           const Vector &binormal,
                           const Vector &normal,
                           const float u0,
                           const float u1,
                           const float u2,
                           Vector &wi,
                           float &pdf,
                           bool &delta) const;

  protected:
    Spectrum mAlbedo;
    Spectrum mAlbedoOverPi;
}; // end NullScatteringBase

#include "NullScatteringBase.inl"

