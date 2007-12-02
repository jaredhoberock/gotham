/*! \file ShaderApi.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to the api used by shaders to
 *         create ScatteringDistributionFunctions.
 */

#ifndef SHADER_API_H
#define SHADER_API_H

#include "../shading/exportShading.h"

#ifdef WIN32
class DLLAPI ShaderApi;
#endif // WIN32

class ScatteringDistributionFunction;
class Spectrum;
class Point;
class Vector;

// XXX TODO make these not static, and have the ShaderApi
//          have a state and also have a FunctionAllocator member
//          then make the ShaderApi a member of Renderer
//          might want to think of a different name at that point?
class ShaderApi
{
  public:
    static ScatteringDistributionFunction *null(void);

    static ScatteringDistributionFunction *diffuse(const Spectrum &Kd);

    static ScatteringDistributionFunction *glossy(const Spectrum &Kr,
                                                  const float eta,
                                                  float exponent);

    static ScatteringDistributionFunction *glossy(const Spectrum &Kr,
                                                  const float eta,
                                                  float uExponent,
                                                  float vExponent);

    static ScatteringDistributionFunction *glossyRefraction(const Spectrum &Kt,
                                                            const float etai,
                                                            const float etat,
                                                            const float exponent);


    static ScatteringDistributionFunction *glass(const float eta,
                                                 const Spectrum &Kr,
                                                 const Spectrum &Kt);

    static ScatteringDistributionFunction *thinGlass(const float eta,
                                                     const Spectrum &Kr,
                                                     const Spectrum &Kt);

    static ScatteringDistributionFunction *mirror(const Spectrum &Kr,
                                                  const float eta);

    static ScatteringDistributionFunction *refraction(const Spectrum &Kt,
                                                      const float etai,
                                                      const float etat);

    static ScatteringDistributionFunction *transparent(const Spectrum &Kt);

    static ScatteringDistributionFunction *uber(const Spectrum &Kd,
                                                const Spectrum &Ks,
                                                const float uShininess,
                                                const float vShininess);

    static ScatteringDistributionFunction *uber(const Spectrum &Kd,
                                                const Spectrum &Ks,
                                                const float shininess);

    static ScatteringDistributionFunction *perspectiveSensor(const Spectrum &Ks,
                                                             const float aspect,
                                                             const Point &origin,
                                                             const Vector &right,
                                                             const Vector &up);

    static ScatteringDistributionFunction *hemisphericalEmission(const Spectrum &Ke);
}; // end ShaderApi

#endif // SHADER_API_H

