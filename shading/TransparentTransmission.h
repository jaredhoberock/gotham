/*! \file TransparentTransmission.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScatteringDistributionFunction
 *         that implements a "pass-through" transmission function.
 */

#ifndef TRANSPARENT_TRANSMISSION_H
#define TRANSPARENT_TRANSMISSION_H

#include "DeltaDistributionFunction.h"

class TransparentTransmission
  : public DeltaDistributionFunction
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef DeltaDistributionFunction Parent;

    /*! Constructor accepts a transmittance.
     *  \param transmittance Sets mTransmittance.
     */
    inline TransparentTransmission(const Spectrum &transmittance);

    /*! This method evaluates a TransparentTransmission function.
     *  \param wi A vector pointing towards the direction of incoming radiance.
     *  \param dg The DifferentialGeometry at the surface point of interest.
     *  \param wo A vector pointing towards the viewing direction.
     *  \return The scattering in direction wo.
     */
    using Parent::evaluate;
    virtual Spectrum evaluate(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi) const;

    /*! This method samples this TransparentTransmission distribution function given a wo,
     *  DifferentialGeometry, and three numbers in the unit interval.
     *  \param wo The direction of scattering.
     *  \param dg The DifferentialGeometry at the point of interest.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param wi The direction of scattering is returned here.
     *  \param pdf The value of the pdf at (u0,u1,u2) is returned here.
     *  \param delta This is set to true; TransparentTransmission is a Dirac delta function.
     *  \param component This is set to 0.
     *  \return The bidirectional scattering from wi to wo is returned here.
     */
    using Parent::sample;
    virtual Spectrum sample(const Vector &wo,
                            const DifferentialGeometry &dg,
                            const float u0,
                            const float u1,
                            const float u2,
                            Vector &wi,
                            float &pdf,
                            bool &delta,
                            ComponentIndex &component) const;

  protected:
    Spectrum mTransmittance;
}; // end TransparentTransmission

#include "TransparentTransmission.inl"

#endif // TRANSPARENT_TRANSMISSION_H
