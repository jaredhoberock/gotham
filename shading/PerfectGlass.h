/*! \file PerfectGlass.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class
 *         implementing a ScatteringDistributionFunction
 *         for the special case of perfect SpecularReflection + SpecularTransmission.
 */

#ifndef PERFECT_GLASS_H
#define PERFECT_GLASS_H

#include "DeltaDistributionFunction.h"
#include "Fresnel.h"

class PerfectGlass
  : public DeltaDistributionFunction
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef DeltaDistributionFunction Parent;

    /*! This constructor creates a PerfectGlass dielectric.
     *  \param r Sets mReflectance.
     *  \param t Sets mTransmittance.
     *  \param etai Sets the index of refraction of the space surrounding the dielectric.
     *  \param etat Sets the index of refraction of the Fresnel dielectric medium.
     */
    PerfectGlass(const Spectrum &r,
                 const Spectrum &t,
                 const float etai, const float etat);

    /*! This method samples this PerfectGlass distribution function given a wo,
     *  DifferentialGeometry, and three numbers in the unit interval.
     *  \param wo The direction of scattering.
     *  \param dg The DifferentialGeometry at the point of interest.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param wi The direction of scattering is returned here.
     *  \param pdf The value of the pdf at (u0,u1,u2) is returned here.
     *  \param delta This is set to true if wi was sampled from a delta
     *               distribution; false, otherwise.
     *  \param component This is set to 0 when wi is sampled from reflection;
     *                   1, when wi is sampled from transmission.
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

    using Parent::evaluatePdf;
    virtual float evaluatePdf(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi,
                              const bool delta,
                              const ComponentIndex component) const;

    /*! This method computes the value of this PerfectGlass and its pdf.
     *  \param wo A Vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the scattering Point of interest.
     *  \param wi A Vector pointing towards the direction of incidence.
     *  \param delta Ignored.  PerfectGlass is always a delta function.
     *  \param component Specifies whether reflection or transmission is of interest.
     *  \param pdf The value of this PefectGlass's pdf is returned here.
     *  \return The value of this PerfectGlass along (wo,dg,wi).
     */
    using Parent::evaluate;
    virtual Spectrum evaluate(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi,
                              const bool delta,
                              const ComponentIndex component,
                              float &pdf) const;

  protected:
    Spectrum sampleReflectance(const Vector &wo,
                               const DifferentialGeometry &dg,
                               Vector &wi) const;

    Spectrum evaluateReflectance(const Vector &wo,
                                 const DifferentialGeometry &dg) const;

    virtual Spectrum sampleTransmittance(const Vector &wo,
                                         const DifferentialGeometry &dg,
                                         Vector &wi) const;

    virtual Spectrum evaluateTransmittance(const Vector &wo,
                                           const DifferentialGeometry &dg) const;
                          
    /*! The reflectance.
     */
    Spectrum mReflectance;

    /*! The transmittance.
     */
    Spectrum mTransmittance;

    /*! The Fresnel dielectric.
     */
    FresnelDielectric mFresnel;
}; // end PerfectGlass

#endif // PERFECT_GLASS_H

