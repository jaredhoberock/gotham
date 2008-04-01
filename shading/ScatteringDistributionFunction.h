/*! \file ScatteringDistributionFunction.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class abstracting
 *         a distribution function over scattering events.
 */

#ifndef SCATTERING_DISTRIBUTION_FUNCTION_H
#define SCATTERING_DISTRIBUTION_FUNCTION_H

#include "../geometry/Vector.h"
#include "../geometry/Normal.h"
#include "../geometry/DifferentialGeometry.h"
#include "FunctionAllocator.h"
#include <spectrum/Spectrum.h>

#ifndef PI
#define PI 3.14159265f 
#endif // PI

#ifndef INV_PI
#define INV_PI (1.0f / PI)
#endif // INV_PI

#ifndef INV_TWOPI
#define INV_TWOPI (1.0f / (2.0f * PI))
#endif // INV_TWOPI

class ScatteringDistributionFunction
{
  public:
    /*! \typedef ComponentIndex
     *  \brief Shorthand.
     */
    typedef size_t ComponentIndex;

    /*! Null destructor does nothing.
     */
    inline virtual ~ScatteringDistributionFunction(void);

    /*! All child classes must implement this method which evaluates a
     *  bidirectional scattering event in an outgoing direction of interest
     *  given a direction of incidence.
     *  \param wo A vector pointing towards the outgoing scattered direction
     *  \param dg The DifferentialGeometry at the Point to be shaded.
     *  \param wi A vector pointing towards the incoming direction.
     *  \return The bidirectional scattering toward direction wo from wi.
     *  \note The default implementation of this method returns
     *        Spectrum::black().
     */
    virtual Spectrum evaluate(const Vector3 &wo,
                              const DifferentialGeometry &dg,
                              const Vector3 &wi) const;

    /*! operator()() method calls evaluate().
     *  \param wo A vector pointing towards the outgoing scattered direction
     *  \param dg The DifferentialGeometry at the Point to be shaded.
     *  \param wi A vector pointing towards the incoming direction.
     *  \return The bidirectional scattering toward direction wo from wi.
     */
    inline Spectrum operator()(const Vector3 &wo,
                               const DifferentialGeometry &dg,
                               const Vector3 &wi) const;

    /*! All child classes must implement this method which evaluates a
     *  scattering event in an outgoing direction of interest.
     *  \param w A vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the Point of interest.
     *  \return The unidirectional scattering in direction we.
     *  \note The default implementation returns Spectrum::black().
     */
    virtual Spectrum evaluate(const Vector3 &w,
                              const DifferentialGeometry &dg) const;

    /*! operator()() method calls evaluate().
     *  \param w A vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the Point of interest.
     *  \return The radiance emitted in direction w.
     */
    inline Spectrum operator()(const Vector3 &w,
                               const DifferentialGeometry &dg) const;

    /*! All child classes must implement this method which samples this
     *  ScatteringDistributionFunction given a wo, DifferentialGeometry,
     *  and three numbers in the unit interval.
     *  \param wo The direction of scattering.
     *  \param dg The DifferentialGeometry at the point of interest.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param wi The direction of scattering is returned here.
     *  \param pdf The value of the pdf at (u0,u1,u2) is returned here.
     *  \param delta This is set to true if wi was sampled from a delta
     *               distribution; false, otherwise.
     *  \param component This is set to the index of the component function which
     *                   generated wi.
     *  \return The bidirectional scattering from wi to wo is returned here.
     */
    virtual Spectrum sample(const Vector3 &wo,
                            const DifferentialGeometry &dg,
                            const float u0,
                            const float u1,
                            const float u2,
                            Vector3 &wi,
                            float &pdf,
                            bool &deltaDistribution,
                            ComponentIndex &component) const;

    /*! All child classes must implement this method which returns the value of
     *  this ScatteringDistributionFunction's pdf given a wo, DifferentialGeometry, and wi.
     *  \param wo A Vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the scattering Point of interest.
     *  \param wi A Vector pointing towards the direction of incidence.
     *  \return The value of the pdf at (wi,dg,wo).
     */
    virtual float evaluatePdf(const Vector3 &wo,
                              const DifferentialGeometry &dg,
                              const Vector3 &wi) const;

    /*! This method evaluates the value of this ScatteringDistributionFunction's pdf given a
     *  wo, DifferentialGeometry, and wi.
     *  This method is included to allow bidirectional path tracing's computation of
     *  MIS weights to work with composite scattering functions.
     *  \param wo A Vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the scattering Point of interest.
     *  \param wi A Vector pointing towards the direction of incidence.
     *  \param delta Whether or not (wo,dg,wi) is known to be a delta function (specular bounce).
     *         If so, this method will include the probability of choosing a specular
     *         bounce into the returned pdf.
     *  \param component The index of the component the bounce (wo,dg,wi) is known to be sampled
     *         from.  Important for specular bounces.
     *  \note The default implementation of this method ignores delta and returns
     *        evaluatePdf(wo,dg,wi).
     */
    virtual float evaluatePdf(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi,
                              const bool delta,
                              const ComponentIndex component) const;

    /*! This method evaluates the value of this ScatteringDistributionFunction and its pdf given a
     *  wo, DifferentialGeometry, and wi.
     *  This method is included to allow bidirectional path tracing's computation of
     *  MIS weights to work with composite scattering functions.
     *  \param wo A Vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the scattering Point of interest.
     *  \param wi A Vector pointing towards the direction of incidence.
     *  \param delta Whether or not (wo,dg,wi) is known to be a delta function (specular bounce).
     *         If so, this method will include the probability of choosing a specular
     *         bounce into the returned pdf.
     *  \param component The index of the component the bounce (wo,dg,wi) is known to be sampled
     *         from.  Important for specular bounces.
     *  \param pdf The value of this ScatteringDistributionFunction's pdf is returned here.
     *  \return The value of this ScatteringDistributionFunction.
     *  \note The default implementation of this method inefficiently
     *        (and incorrectly for delta functions) calls evaluatePdf() and then
     *        the other version of evaluate().
     */
    virtual Spectrum evaluate(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi,
                              const bool delta,
                              const ComponentIndex component,
                              float &pdf) const;

    /*! This method evaluates the solid angle pdf of the
     *  direction of interest.
     *  \param w The direction of interest.
     *  \param dg The DifferentialGeometry at the Point of interest.
     *  \return The solid angle pdf of w.
     */
    virtual float evaluatePdf(const Vector3 &w,
                              const DifferentialGeometry &dg) const;

    /*! This method inverts this ScatteringDistributionFunction's mapping
     *  from a direction to the unit square.
     *  \param w The direction of interest.
     *  \param dg the DifferentialGeometry at the Point of interest.
     *  \param u0 The first coordinate of the corresponding point in the unit
     *            square is returned here.
     *  \param u1 The second coordinate of the corresponding point in the unit
     *            square is returned here.
     */
    virtual void invert(const Vector &w,
                        const DifferentialGeometry &dg,
                        float &u0,
                        float &u1) const;

    /*! All child classes must implement this method which samples this
     *  ScatteringDistributionFunction given a DifferentialGeometry,
     *  and three numbers in the unit interval.
     *  \param dg The DifferentialGeometry at the point of interest.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param w The direction of scattering is returned here.
     *  \param pdf The value of the pdf at (u0,u1,u2) is returned here.
     *  \param delta This is set to true if wi was sampled from a delta
     *               distribution; false, otherwise.
     *  \return The unidirectional scattering to w is returned here.
     */
    virtual Spectrum sample(const DifferentialGeometry &dg,
                            const float u0,
                            const float u1,
                            const float u2,
                            Vector3 &w,
                            float &pdf,
                            bool &delta) const;

    /*! This static method evaluates whether or not an incident and exitant direction
     *  are in the same hemisphere with respect to the Normal direction.
     *  \param wi A vector pointing towards the direction of incident radiance.
     *  \param n The surface Normal at the point of interest.
     *  \param wo A vector pointing towards the direction of exitant radiance.
     *  \return true if wi and wo are in the same hemisphere with respect to
     *          n; false, otherwise.
     */
    inline static bool areSameHemisphere(const Vector3 &wo,
                                         const Normal &n,
                                         const Vector3 &wi);

    inline static bool areSameHemisphere(const float coso,
                                         const float cosi);

    /*! This method indicates whether or not this ScatteringDistributionFunction
     *  is completely described by one or more Dirac delta functions.
     *  \return true if this ScatteringDistributionFunction is completely described as a
     *          sum of one or more Dirac delta functions; false, otherwise.
     *  \note The default implementation of this method returns false.
     */
    virtual bool isSpecular(void) const;

    /*! Overload the new operator.
     */
    void *operator new(size_t size, FunctionAllocator &alloc);

    /*! This method clones this ScatteringDistributionFunction.
     *  \param allocator The FunctionAllocator to allocate from.
     *  \return a pointer to a newly-allocated clone of this ScatteringDistributionFunction; 0,
     *          if no memory could be allocated.
     */
    virtual ScatteringDistributionFunction *clone(FunctionAllocator &allocator) const;

  protected:
    /*! This method evaluates the geometric term common to microfacet scattering models.
     *  \param nDotWo
     *  \param nDotWi
     *  \param nDotWh
     *  \param woDotWh
     *  \return The geometric term.
     */
    static float evaluateGeometricTerm(const float nDotWo,
                                       const float nDotWi,
                                       const float nDotWh,
                                       const float woDotWh);
}; // end ScatteringDistributionFunction

#include "ScatteringDistributionFunction.inl"

#endif // SCATTERING_DISTRIBUTION_FUNCTION_H

