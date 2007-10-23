/*! \file SpecularTransmission.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a BSDF
 *         for specular transmission.
 */

#ifndef SPECULAR_TRANSMISSION_H
#define SPECULAR_TRANSMISSION_H

#include <spectrum/Spectrum.h>
#include "DeltaDistributionFunction.h"
#include "Fresnel.h"

class SpecularTransmission
  : public DeltaDistributionFunction
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef DeltaDistributionFunction Parent;

    /*! Constructor accepts a transmission and indices of refraction.
     *  \param t The transmittance of this SpecularTransmission.
     *  \param etai The index of refraction on the outside of the interface
     *              of this SpecularTransmission.
     *  \param etat The index of refraction on the inside of the interface
     *              of this SpecularTransmission.
     */
    SpecularTransmission(const Spectrum &t,
                         const float etai, const float etat);

    /*! This method evaluates this SpecularTransmission function.
     *  \param wi A vector pointing towards the direction of incoming radiance.
     *  \param dg The DifferentialGeometry at the surface point of interest.
     *  \param wo A vector pointing towards the viewing direction.
     *  \return The scattering in direction wo.
     */
    using Parent::evaluate;
    virtual Spectrum evaluate(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi) const;

    /*! This method samples this SpecularTransmission function given a wo,
     *  DifferentialGeometry, and three numbers in the unit interval.
     *  \param wo The direction of scattering.
     *  \param dg The DifferentialGeometry at the point of interest.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param wi The direction of scattering is returned here.
     *  \param pdf The value of the pdf at (u0,u1,u2) is returned here.
     *  \param delta This is set to true.
     *  \param index This is set to 0.
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
                            ComponentIndex &index) const;

    /*! This method computes the value of this SpecularTransmission and its pdf.
     *  \param wo A Vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the scattering Point of interest.
     *  \param wi A Vector pointing towards the direction of incidence.
     *  \param delta When this is true, 
     *  \param component Ignored.  SpecularTransmission has only one component.
     *  \param pdf The value of this SpecularTransmission's pdf is returned here.
     *  \return The value of this SpecularTransmission along (wo,dg,wi).
     */
    virtual Spectrum evaluate(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi,
                              const bool delta,
                              const ComponentIndex component,
                              float &pdf) const;

  protected:
    /*! The transmittance.
     */
    Spectrum mTransmittance;

    /*! The Fresnel dielectric.
     */
    FresnelDielectric mFresnel;
}; // end SpecularTransmission

#endif // SPECULAR_TRANSMISSION_H

