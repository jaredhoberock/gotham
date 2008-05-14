/*! \file PhongTransmission.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class implementing
 *         glossy transmission.
 */

#ifndef PHONG_TRANSMISSION_H
#define PHONG_TRANSMISSION_H

#include "ScatteringDistributionFunction.h"
#include "Fresnel.h"

class PhongTransmission
  : public ScatteringDistributionFunction
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScatteringDistributionFunction Parent;

    /*! Constructor accepts a transmission, indices of refraction,
     *  and a Phong exponent.
     *  \param t The transmittance of this PhongTransmission.
     *  \param etai The index of refraction on the outside of the interface
     *              of this PhongTransmission.
     *  \param etat The index of refraction on the inside of the interface
     *              of this PhongTransmission.
     *  \param exponent The Phong exponent of this PhongTransmission.
     */
    PhongTransmission(const Spectrum &t,
                      const float etai, const float etat,
                      const float exponent);

    /*! This method evaluates this PhongTransmission function.
     *  \param wi A vector pointing towards the direction of incoming radiance.
     *  \param dg The DifferentialGeometry at the surface point of interest.
     *  \param wo A vector pointing towards the viewing direction.
     *  \return The scattering in direction wo.
     */
    using Parent::evaluate;
    virtual Spectrum evaluate(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi) const;

    /*! This method evaluates the value of this PhongTransmission and its pdf given a
     *  wo, DifferentialGeometry, and wi.
     *  This method is included to allow bidirectional path tracing's computation of
     *  MIS weights to work with composite scattering functions.
     *  \param wo A Vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the scattering Point of interest.
     *  \param wi A Vector pointing towards the direction of incidence.
     *  \param delta Ignored.  PhongTransmission is not a delta function.
     *  \param component Ignored.  PhongTransmission is a single component.
     *  \param pdf The value of this PhongTransmission's pdf is returned here.
     *  \return The value of this PhongTransmission.
     */
    virtual Spectrum evaluate(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi,
                              const bool delta,
                              const ComponentIndex component,
                              float &pdf) const;

    /*! This method samples this PhongTransmission function given a wo,
     *  DifferentialGeometry, and three numbers in the unit interval.
     *  \param wo The direction of scattering.
     *  \param dg The DifferentialGeometry at the point of interest.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param wi The direction of scattering is returned here.
     *  \param pdf The value of the pdf at (u0,u1,u2) is returned here.
     *  \param delta This is set to false.
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
                            ComponentIndex &index) const;

    /*! This method returns the value of
     *  this PhongTransmission's pdf given a wo, DifferentialGeometry, and wi.
     *  \param wo A Vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the scattering Point of interest.
     *  \param wi A Vector pointing towards the direction of incidence.
     *  \return The value of the pdf at (wi,dg,wo).
     */
    using Parent::evaluatePdf;
    virtual float evaluatePdf(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi) const;

  protected:
    /*! The transmittance.
     */
    Spectrum mTransmittance;

    /*! The transmittance divided by 2PI.
     */
    Spectrum mTransmittanceOverTwoPi;

    /*! The Phong exponent.
     */
    float mExponent;

    /*! The Fresnel dielectric.
     */
    FresnelDielectric mFresnel;
}; // end PhongTransmission

#endif // PHONG_TRANSMISSION_H

