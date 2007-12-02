/*! \file stdmaterial.h
 *  \author Jared Hoberock
 *  \brief Thunk functions for hiding ShaderApi nastiness from shaders.
 */

#include "../api/ShaderApi.h"

inline ScatteringDistributionFunction *diffuse(const Spectrum &Kd)
{
  return ShaderApi::diffuse(Kd);
} // end ScatteringDistributionFunction::diffuse()

inline ScatteringDistributionFunction *glossy(const Spectrum &Kr,
                                              const float eta,
                                              const float exponent)
{
  return ShaderApi::glossy(Kr, eta, exponent);
} // end ScatteringDistributionFunction::glossy()

inline ScatteringDistributionFunction *glossy(const Spectrum &Kr,
                                              const float eta,
                                              const float uExponent,
                                              const float vExponent)
{
  return ShaderApi::glossy(Kr, eta, uExponent, vExponent);
} // end ScatteringDistributionFunction::glossy()

inline ScatteringDistributionFunction *glossyRefraction(const Spectrum &Kt,
                                                        const float etai,
                                                        const float etat,
                                                        const float exponent)
{
  return ShaderApi::glossyRefraction(Kt, etai, etat, exponent);
} // end glossyRefraction()

inline ScatteringDistributionFunction *glass(const float eta,
                                             const Spectrum &Kr,
                                             const Spectrum &Kt)
{
  return ShaderApi::glass(eta, Kr, Kt);
} // end glass()

inline ScatteringDistributionFunction *thinGlass(const float eta,
                                                 const Spectrum &Kr,
                                                 const Spectrum &Kt)
{
  return ShaderApi::thinGlass(eta, Kr, Kt);
} // end thinGlass()

inline ScatteringDistributionFunction *mirror(const Spectrum &Kr,
                                              const float eta)
{
  return ShaderApi::mirror(Kr, eta);
} // end mirror()

inline ScatteringDistributionFunction *refraction(const Spectrum &Kr,
                                                  const float etai,
                                                  const float etat)
{
  return ShaderApi::refraction(Kr, etai, etat);
} // end refraction()

inline ScatteringDistributionFunction *transparent(const Spectrum &Kt)
{
  return ShaderApi::transparent(Kt);
} // end transparent()

inline ScatteringDistributionFunction *uber(const Spectrum &Kd,
                                            const Spectrum &Ks,
                                            const float uShininess,
                                            const float vShininess)
{
  return ShaderApi::uber(Kd, Ks, uShininess, vShininess);
} // end uber()

inline ScatteringDistributionFunction *uber(const Spectrum &Kd,
                                            const Spectrum &Ks,
                                            const float shininess)
{
  return ShaderApi::uber(Kd, Ks, shininess);
} // end uber()

inline ScatteringDistributionFunction *perspectiveSensor(const Spectrum &Ks,
                                                         const float aspect,
                                                         const Point &origin,
                                                         const Vector &right,
                                                         const Vector &up)
{
  return ShaderApi::perspectiveSensor(Ks, aspect, origin, right, up);
} // end perspective()

inline ScatteringDistributionFunction *hemisphericalEmission(const Spectrum &Ke)
{
  return ShaderApi::hemisphericalEmission(Ke);
} // end hemisphericalEmission()

