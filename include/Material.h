/*! \file Material.h
 *  \author Jared Hoberock
 *  \brief Material class for gotham renderer.
 */

#ifndef MATERIAL_H
#define MATERIAL_H

#include "exportShading.h"

// XXX don't require another #include here for DifferentialGeometry
//class DifferentialGeometry;
#include "../geometry/DifferentialGeometry.h"
class ShadingInterface;
class ScatteringDistributionFunction;

class Material
{
  public:
    virtual ~Material(void);
    virtual ScatteringDistributionFunction *evaluateScattering(ShadingInterface &context, const DifferentialGeometry &dg) const;
    virtual ScatteringDistributionFunction *evaluateEmission(ShadingInterface &context, const DifferentialGeometry &dg) const;
    virtual ScatteringDistributionFunction *evaluateSensor(ShadingInterface &context, const DifferentialGeometry &dg) const;
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

