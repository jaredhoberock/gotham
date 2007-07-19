/*! \file Material.h
 *  \author Jared Hoberock
 *  \brief Material class for gotham renderer.
 */

#ifndef MATERIAL_H
#define MATERIAL_H

#include "../geometry/DifferentialGeometry.h"
class ScatteringDistributionFunction;

class Material
{
  public:
    virtual ~Material(void);
    virtual ScatteringDistributionFunction *evaluateScattering(const DifferentialGeometry &dg) const;
    virtual ScatteringDistributionFunction *evaluateEmission(const DifferentialGeometry &dg) const;
    virtual ScatteringDistributionFunction *evaluateSensor(const DifferentialGeometry &dg) const;
    virtual const char *getName(void) const;

    /*! XXX Is there a more elegant way to do this?
     *  This method indicates whether or not this Material
     *  implements evaluateEmission() as a hint to importance
     *  sampling.
     *  \return true if evaluateEmission() is implemented to
     *               return an EmissionFunction; false, otherwise.
     */
    virtual bool isEmitter(void) const;

    /*! This method indicates whether or not this Material
     *  implements evaluateSensor() as a hint to importance
     *  sampling.
     *  \return true if evaluateSensor() is implemented to
     *          return a SensorFunction; false, otherwise.
     */
    virtual bool isSensor(void) const;
}; // end Material

#endif // MATERIAL_H

