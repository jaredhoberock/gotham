/*! \file ScatteringDistributionFunction.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class abstracting
 *         a distribution function over scattering events.
 */

#ifndef SCATTERING_DISTRIBUTION_FUNCTION_H
#define SCATTERING_DISTRIBUTION_FUNCTION_H

#include <spectrum/Spectrum.h>
#include "../geometry/Vector.h"
#include "../geometry/Normal.h"
#include "../geometry/DifferentialGeometry.h"
#include "FunctionAllocator.h"

#ifndef PI
#define PI 3.14159265f 
#endif // PI

#ifndef INV_PI
#define INV_PI (1.0f / PI)
#endif // INV_PI

class ScatteringDistributionFunction
{
  public:
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
     *  \return The bidirectional scattering from wi to wo is returned here.
     */
    virtual Spectrum sample(const Vector3 &wo,
                            const DifferentialGeometry &dg,
                            const float u0,
                            const float u1,
                            const float u2,
                            Vector3 &wi,
                            float &pdf) const;

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

    /*! This method evaluates the solid angle pdf of the
     *  sensing direction of interest.
     *  \param w The sensing direction of interest.
     *  \param dg The DifferentialGeometry at the Point of interest.
     *  \return The solid angle pdf of w.
     */
    virtual float evaluatePdf(const Vector3 &w,
                              const DifferentialGeometry &dg) const;

    /*! All child classes must implement this method which samples this
     *  ScatteringDistributionFunction given a DifferentialGeometry,
     *  and three numbers in the unit interval.
     *  \param dg The DifferentialGeometry at the point of interest.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param w The direction of scattering is returned here.
     *  \param pdf The value of the pdf at (u0,u1,u2) is returned here.
     *  \return The unidirectional scattering to w is returned here.
     */
    virtual Spectrum sample(const DifferentialGeometry &dg,
                            const float u0,
                            const float u1,
                            const float u2,
                            Vector3 &w,
                            float &pdf) const;

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

    static FunctionAllocator mPool;

    /*! Overload the new operator.
     */
    void *operator new(unsigned int size);
}; // end ScatteringDistributionFunction

#include "ScatteringDistributionFunction.inl"

#endif // SCATTERING_DISTRIBUTION_FUNCTION_H

