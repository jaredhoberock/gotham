/*! \file Material.h
 *  \author Jared Hoberock
 *  \brief Material class for gotham renderer.
 */

#ifndef MATERIAL_H
#define MATERIAL_H

#include "exportShading.h"

// XXX don't require another #include here for DifferentialGeometry
//class DifferentialGeometry;
#include "DifferentialGeometry.h"
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

    /*! This method returns the source of this Material.
     *  \return A string containing the source of the shader describing
     *          this Material.
     */
    virtual const char *getSource(void) const;

    /*! This method returns the size in bytes of this Material's
     *  parameters.
     *  \return As above.
     */
    virtual size_t getScatteringParametersSize(void) const;

    /*! This method copies this Material's scattering parameters to
     *  location given by the pointer.
     *  \param ptr The location to copy to.
     */
    virtual void getScatteringParameters(void *ptr) const;

    /*! This method returns the size in bytes of this Material's
     *  parameters.
     *  \return As above.
     */
    virtual size_t getEmissionParametersSize(void) const;

    /*! This method copies this Material's emission parameters to
     *  location given by the pointer.
     *  \param ptr The location to copy to.
     */
    virtual void getEmissionParameters(void *ptr) const;

    /*! This method returns the size in bytes of this Material's
     *  parameters.
     *  \return As above.
     */
    virtual size_t getSensorParametersSize(void) const;

    /*! This method copies this Material's sensor parameters to
     *  location given by the pointer.
     *  \param ptr The location to copy to.
     */
    virtual void getSensorParameters(void *ptr) const;

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

