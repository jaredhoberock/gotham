/*! \file CompositeBase.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class
 *         encapsulating a BSDF comprised of a
 *         composition of primitive BSDFs.
 */

#pragma once

template<typename V3, typename S3,
         typename F0, typename F1>
  class CompositeBase
{
  public:
    typedef V3 Vector;
    typedef S3 Spectrum;
    static const unsigned int NUM_COMPONENTS = 2;

    inline CompositeBase(const F0 &c0, const F1 &c1);

    inline Spectrum evaluate(const Vector &wo,
                             const Vector &normal,
                             const Vector &wi) const;

    /*! This method samples this CompositeBase given a Wo,
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
     *  \param delta This is set to true if the sampled component is a delta function.
     *  \param component This is set to the index of the sampled component.
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
    F0 mComponent0;
    F1 mComponent1;
}; // end CompositeBase

#include "CompositeBase.inl"

