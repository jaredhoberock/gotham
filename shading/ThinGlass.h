/*! \file ThinGlass.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a PerfectGlass scattering function
 *         whose medium is very thin relative to the size of the world.
 *         Essentially, this is glass with no refraction.
 */

#ifndef THIN_GLASS_H
#define THIN_GLASS_H

#include "PerfectGlass.h"

class ThinGlass
  : public PerfectGlass
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef PerfectGlass Parent;

    /*! This constructor creates a ThinGlass dielectric.
     *  \param r Sets mReflectance.
     *  \param t Sets mTransmittance.
     *  \param etai Sets the index of refraction of the space surrounding the dielectric.
     *  \param etat Sets the index of refraction of the Fresnel dielectric medium.
     */
    ThinGlass(const Spectrum &r,
              const Spectrum &t,
              const float etai,
              const float etat);

  protected:
    virtual Spectrum sampleTransmittance(const Vector &wo,
                                         const DifferentialGeometry &dg,
                                         Vector &wi) const;

    virtual Spectrum evaluateTransmittance(const Vector &wo,
                                           const DifferentialGeometry &dg) const;
}; // end ThinGlass

#endif // THIN_GLASS_H

