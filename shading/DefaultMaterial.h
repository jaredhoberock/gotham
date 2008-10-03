/*! \file DefaultMaterial.h
 *  \author Jared Hoberock
 *  \brief This class serves as a default Material
 *         with Lambertian scattering that we are guaranteed
 *         always exists.
 */

#ifndef DEFAULT_MATERIAL_H
#define DEFAULT_MATERIAL_H

#include "../include/detail/Material.h"

class DefaultMaterial
  : public Material
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Material Parent;

    virtual const char *getName(void) const;
    virtual const char *getSource(void) const;
    virtual ScatteringDistributionFunction *evaluateScattering(ShadingInterface &context, const DifferentialGeometry &dg) const;
}; // end DefaultMaterial

#endif // DEFAULT_MATERIAL_H

