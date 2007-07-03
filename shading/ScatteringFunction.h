/*! \file ScatteringFunction
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScatteringFunction class.
 */

#ifndef SCATTERING_FUNCTION_H
#define SCATTERING_FUNCTION_H

#include <spectrum/Spectrum.h>
#include "../geometry/Vector.h"

#ifndef PI
#define PI 3.14159265f 
#endif // PI

class ScatteringFunction
{
  public:
    /*! All child classes must implement this method which evaluates the
     *  scattering in an outgoing direction of interest given a direction of
     *  incidence.
     *  \param wi A vector pointing towards the incoming direction.
     *  \param wo A vector pointing towards the outgoing scattered direction
     *  \return The scattering toward direction wo from wi.
     */
    virtual Spectrum evaluate(const Vector3 &wi,
                              const Vector3 &wo) const = 0;

    /*! This static method evaluates whether or not an incident and exitant direction
     *  are in the same hemisphere with respect to the canonical (0,0,1)
     *  direction.
     *  \param wi A vector pointing towards the direction of incident radiance.
     *  \param wo A vector pointing towards the direction of exitant radiance.
     *  \return true if wi and wo are in the same hemisphere with respect to
     *          (0,0,1); false, otherwise.
     */
    inline static bool areSameHemisphere(const Vector3 &wo,
                                         const Vector3 &wi);
}; // end ScatteringFunction

#include "ScatteringFunction.inl"

#endif // SCATTERING_FUNCTION_H

