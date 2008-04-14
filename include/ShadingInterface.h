/*! \file ShadingInterface.h
 *  \author Jared Hoberock
 *  \brief Defines an abstract interface that
 *         shaders interact with to perform
 *         various common shading functions.
 */

#pragma once

#include "exportShading.h"

#ifdef WIN32
class DLLAPI ShadingInterface;
#endif // WIN32

class ScatteringDistributionFunction;
class Spectrum;
class Point;
class Vector;

class ShadingInterface
{
  public:
    virtual ~ShadingInterface(void){};

    virtual ScatteringDistributionFunction *null(void) = 0;

    virtual ScatteringDistributionFunction *diffuse(const Spectrum &Kd) = 0;

    virtual ScatteringDistributionFunction *glossy(const Spectrum &Kr,
                                                   const float eta,
                                                   float exponent) = 0;

    virtual ScatteringDistributionFunction *glossy(const Spectrum &Kr,
                                                   const float eta,
                                                   float uExponent,
                                                   float vExponent) = 0;

    virtual ScatteringDistributionFunction *glossyRefraction(const Spectrum &Kt,
                                                             const float etai,
                                                             const float etat,
                                                             const float exponent) = 0;


    virtual ScatteringDistributionFunction *glass(const float eta,
                                                  const Spectrum &Kr,
                                                  const Spectrum &Kt) = 0;

    virtual ScatteringDistributionFunction *thinGlass(const float eta,
                                                      const Spectrum &Kr,
                                                      const Spectrum &Kt) = 0;

    virtual ScatteringDistributionFunction *mirror(const Spectrum &Kr,
                                                   const float eta) = 0;

    virtual ScatteringDistributionFunction *refraction(const Spectrum &Kt,
                                                       const float etai,
                                                       const float etat) = 0;

    virtual ScatteringDistributionFunction *transparent(const Spectrum &Kt) = 0;

    virtual ScatteringDistributionFunction *uber(const Spectrum &Kd,
                                                 const Spectrum &Ks,
                                                 const float uShininess,
                                                 const float vShininess) = 0;

    virtual ScatteringDistributionFunction *uber(const Spectrum &Kd,
                                                 const Spectrum &Ks,
                                                 const float shininess) = 0;

    virtual ScatteringDistributionFunction *uber(const Spectrum &Ks,
                                                 const float shininess,
                                                 const Spectrum &Kr,
                                                 float eta) = 0;

    virtual ScatteringDistributionFunction *perspectiveSensor(const Spectrum &Ks,
                                                              const float aspect,
                                                              const Point &origin) = 0;

    virtual ScatteringDistributionFunction *hemisphericalEmission(const Spectrum &Ke) = 0;

    virtual float noise(const float x, const float y, const float z) = 0;
}; // end ShadingInterface

