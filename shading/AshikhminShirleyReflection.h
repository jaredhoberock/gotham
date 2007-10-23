/*! \file AshikhminShirleyReflection.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScatteringDistributionFunction
 *         which implements the anisotropic Ashikhmin-Shirley glossy
 *         reflection model.
 */

#ifndef ASHIKHMIN_SHIRLEY_REFLECTION_H
#define ASHIKHMIN_SHIRLEY_REFLECTION_H

#include "ScatteringDistributionFunction.h"
class Fresnel;

class AshikhminShirleyReflection
  : public ScatteringDistributionFunction
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScatteringDistributionFunction Parent;

    /*! Constructor accepts a reflectance, index of refraction,
     *  and two Ashikhmin-Shirley exponents to create an anisotropic
     *  glossy Fresnel conductor.
     *  \param r The reflectance of this AshikhminShirleyReflection.
     *  \param eta Sets the index of refraction of the Fresnel conductor.
     *  \param uExponent The Ashikhmin-Shirley exponent in the u direction.
     *  \param vExponent The Ashikhmin-Shriley exponent in the v direction.
     */
    AshikhminShirleyReflection(const Spectrum &r,
                               const float eta,
                               const float uExponent,
                               const float vExponent);

    /*! This method evaluates this AshikhminShirleyReflection function.
     *  \param wi A vector pointing towards the direction of incoming radiance.
     *  \param dg The DifferentialGeometry at the surface point of interest.
     *  \param wo A vector pointing towards the viewing direction.
     *  \return The scattering in direction wo.
     */
    using Parent::evaluate;
    virtual Spectrum evaluate(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi) const;

    /*! This method evaluates the value of this AshikhminShirleyReflection and its pdf given a
     *  wo, DifferentialGeometry, and wi.
     *  This method is included to allow bidirectional path tracing's computation of
     *  MIS weights to work with composite scattering functions.
     *  \param wo A Vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the scattering Point of interest.
     *  \param wi A Vector pointing towards the direction of incidence.
     *  \param delta Ignored.  AshikhminShirleyReflection is not a delta function.
     *  \param component Ignored.  AshikhminShirleyReflection is a single component.
     *  \param pdf The value of this AshikhminShirleyReflection's pdf is returned here.
     *  \return The value of this AshikhminShirleyReflection.
     */
    virtual Spectrum evaluate(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi,
                              const bool delta,
                              const ComponentIndex component,
                              float &pdf) const;

    /*! This method samples this AshikhminShirleyReflection function given a wo,
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
                            ComponentIndex &component) const;

    /*! This method returns the value of
     *  this AshikhminShirleyReflection's pdf given a wo, DifferentialGeometry, and wi.
     *  \param wo A Vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the scattering Point of interest.
     *  \param wi A Vector pointing towards the direction of incidence.
     *  \return The value of the pdf at (wi,dg,wo).
     */
    using Parent::evaluatePdf;
    virtual float evaluatePdf(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi) const;

    /*! This method clones this AshikhminShirleyReflection.
     *  \param allocator The FunctionAllocator to allocate from.
     *  \return a pointer to a newly-allocated clone of this AshikhminShirleyReflection; 0,
     *          if no memory could be allocated.
     */
    virtual ScatteringDistributionFunction *clone(FunctionAllocator &allocator) const;

  protected:
    /*! The reflectance.
     */
    Spectrum mReflectance;

    /*! The exponents.
     */
    float mNu;
    float mNv;

    /*! The Fresnel object.
     */
    Fresnel *mFresnel;
}; // end AshikhminShirleyReflection

#endif // ASHIKHMIN_SHIRLEY_REFLECTION_H

