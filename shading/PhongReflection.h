/*! \file PhongReflection.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class implementing
 *         glossy Reflection.
 */

#ifndef PHONG_REFLECTION_H
#define PHONG_REFLECTION_H

#include <gpcpu/Vector.h>
#include "ScatteringDistributionFunction.h"
#include "functions/PhongReflectionBase.h"

class PhongReflection
  : public ScatteringDistributionFunction,
    public PhongReflectionBase<Vector,Spectrum>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    //typedef ScatteringDistributionFunction Parent;
    typedef ScatteringDistributionFunction Parent0;

    typedef PhongReflectionBase<Vector,Spectrum> Parent1;

    /*! Constructor accepts a reflectance, index of refraction,
     *  and a Phong exponent to create a glossy Fresnel conductor.
     *  \param r The reflectivity of this PhongReflection.
     *  \param eta Sets the index of refraction of the Fresnel conductor.
     *  \param exponent The Phong exponent of this PhongReflection.
     */
    PhongReflection(const Spectrum &r,
                    const float eta,
                    const float exponent);

    /*! Constructor accepts a reflectance, indices of refraction,
     *  and a Phong exponent to create a glossy Fresnel dielectric.
     *  \param t The transmittance of this PhongReflection.
     *  \param etai Sets the index of refraction of the space surrounding the dielectric.
     *  \param etat Sets the index of refraction of the Fresnel dielectric medium.
     *  \param exponent The Phong exponent of this PhongReflection.
     *  \param alloc A FunctionAllocator object for allocating the Fresnel object.
     */
    PhongReflection(const Spectrum &t,
                    const float etai,
                    const float etat,
                    const float exponent);

    /*! This method evaluates this PhongReflection function.
     *  \param wi A vector pointing towards the direction of incoming radiance.
     *  \param dg The DifferentialGeometry at the surface point of interest.
     *  \param wo A vector pointing towards the viewing direction.
     *  \return The scattering in direction wo.
     */
    using Parent0::evaluate;
    virtual Spectrum evaluate(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi) const;

    /*! This method samples this PhongReflection function given a wo,
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
    using Parent0::sample;
    virtual Spectrum sample(const Vector &wo,
                            const DifferentialGeometry &dg,
                            const float u0,
                            const float u1,
                            const float u2,
                            Vector &wi,
                            float &pdf,
                            bool &delta,
                            ComponentIndex &component) const;

    /*! This method evaluates the value of this PhongReflection and its pdf given a
     *  wo, DifferentialGeometry, and wi.
     *  This method is included to allow bidirectional path tracing's computation of
     *  MIS weights to work with composite scattering functions.
     *  \param wo A Vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the scattering Point of interest.
     *  \param wi A Vector pointing towards the direction of incidence.
     *  \param delta Ignored.  PhongReflection is not a delta function.
     *  \param component Ignored.  PhongReflection is a single component.
     *  \param pdf The value of this PhongReflection's pdf is returned here.
     *  \return The value of this PhongReflection.
     */
    virtual Spectrum evaluate(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi,
                              const bool delta,
                              const ComponentIndex component,
                              float &pdf) const;

    /*! This method returns the value of
     *  this PhongReflection's pdf given a wo, DifferentialGeometry, and wi.
     *  \param wo A Vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the scattering Point of interest.
     *  \param wi A Vector pointing towards the direction of incidence.
     *  \return The value of the pdf at (wi,dg,wo).
     */
    using Parent0::evaluatePdf;
    virtual float evaluatePdf(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi) const;
}; // end PhongReflection

#endif // PHONG_REFLECTION_H

