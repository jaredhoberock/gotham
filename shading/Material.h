/*! \file Material.h
 *  \author Jared Hoberock
 *  \brief Material class for gotham renderer.
 */

#ifndef MATERIAL_H
#define MATERIAL_H

#include "../geometry/DifferentialGeometry.h"
class ScatteringFunction;

class Material
{
  public:
    virtual ScatteringFunction *evaluate(const DifferentialGeometry &dg) const = 0;
    virtual const char *getName(void) const = 0;
}; // end Material

#endif // MATERIAL_H

