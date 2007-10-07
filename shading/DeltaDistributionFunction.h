/*! \file DeltaDistributionFunction.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class of
 *         ScatteringDistributionFunctions whose scattering
 *         function is defined by a Dirac delta function.
 */

#ifndef DELTA_DISTRIBUTION_FUNCTION_H
#define DELTA_DISTRIBUTION_FUNCTION_H

#include "ScatteringDistributionFunction.h"

class DeltaDistributionFunction
  : public ScatteringDistributionFunction
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScatteringDistributionFunction Parent;

    /*! This method returns 0.0f.
     *  \param wo A Vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the scattering Point of interest.
     *  \param wi A Vector pointing towards the direction of incidence.
     *  \return 0.0f.
     */
    using Parent::evaluatePdf;
    virtual float evaluatePdf(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi) const;

    /*! This method returns 1.0f when delta is true; 0.0f, otherwise.
     *  \param wo A Vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the scattering Point of interest.
     *  \param wi A Vector pointing towards the direction of incidence.
     *  \param delta Whether or not (wo,dg,wi) is known to be a delta function (specular bounce).
     *         If so, this method will include the probability of choosing a specular
     *         bounce into the returned pdf.
     *  \param component The index of the component (wo,dg,wi) is known to be sampled from.
     *  \return As above.
     */
    virtual float evaluatePdf(const Vector &wo,
                              const DifferentialGeometry &dg,
                              const Vector &wi,
                              const bool delta,
                              const ComponentIndex component) const;

    /*! This method returns true.
     *  \return true
     */
    virtual bool isSpecular(void) const;
}; // end DeltaDistributionFunction

#endif // DELTA_DISTRIBUTION_FUNCTION_H

