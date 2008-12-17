/*! \file stdmaterial.h
 *  \author Jared Hoberock
 *  \brief Thunk functions for hiding ShadingInterface nastiness from shaders.
 */

#include "detail/ShadingInterface.h"

// The idea here is that we need an allocator for the different
// ScatteringDistributionFunctions
// So we will require that the shader set this at the beginning of its
// function before calling any of these functions.
// XXX This is probably not reentrant
extern ShadingInterface *gContext;

inline ScatteringDistributionFunction *diffuse(const Spectrum &Kd)
{
  return gContext->diffuse(Kd);
} // end ScatteringDistributionFunction::diffuse()

inline ScatteringDistributionFunction *glossy(const Spectrum &Kr,
                                              const float eta,
                                              const float exponent)
{
  return gContext->glossy(Kr, eta, exponent);
} // end ScatteringDistributionFunction::glossy()

inline ScatteringDistributionFunction *glossy(const Spectrum &Kr,
                                              const float etai,
                                              const float etat,
                                              const float exponent)
{
  return gContext->glossy(Kr, etai, etat, exponent);
} // end ScatteringDistributionFunction::glossy()

inline ScatteringDistributionFunction *anisotropic(const Spectrum &Kr,
                                                   const float eta,
                                                   const float uExponent,
                                                   const float vExponent)
{
  return gContext->anisotropic(Kr, eta, uExponent, vExponent);
} // end ScatteringDistributionFunction::glossy()

inline ScatteringDistributionFunction *anisotropic(const Spectrum &Kr,
                                                   const float etai,
                                                   const float etat,
                                                   const float uExponent,
                                                   const float vExponent)
{
  return gContext->anisotropic(Kr, etai, etat, uExponent, vExponent);
} // end ScatteringDistributionFunction::glossy()

inline ScatteringDistributionFunction *glossyRefraction(const Spectrum &Kt,
                                                        const float etai,
                                                        const float etat,
                                                        const float exponent)
{
  return gContext->glossyRefraction(Kt, etai, etat, exponent);
} // end glossyRefraction()

inline ScatteringDistributionFunction *glass(const float eta,
                                             const Spectrum &Kr,
                                             const Spectrum &Kt)
{
  return gContext->glass(eta, Kr, Kt);
} // end glass()

inline ScatteringDistributionFunction *thinGlass(const float eta,
                                                 const Spectrum &Kr,
                                                 const Spectrum &Kt)
{
  return gContext->thinGlass(eta, Kr, Kt);
} // end thinGlass()

inline ScatteringDistributionFunction *mirror(const Spectrum &Kr,
                                              const float eta)
{
  return gContext->mirror(Kr, eta);
} // end mirror()

inline ScatteringDistributionFunction *reflection(const Spectrum &Kr,
                                                  const float etai, 
                                                  const float etat)
{
  return gContext->reflection(Kr,etai,etat);
} // end reflection()

inline ScatteringDistributionFunction *refraction(const Spectrum &Kr,
                                                  const float etai,
                                                  const float etat)
{
  return gContext->refraction(Kr, etai, etat);
} // end refraction()

inline ScatteringDistributionFunction *transparent(const Spectrum &Kt)
{
  return gContext->transparent(Kt);
} // end transparent()

inline ScatteringDistributionFunction *composite(ScatteringDistributionFunction *f0,
                                                 ScatteringDistributionFunction *f1)
{
  return gContext->composite(f0,f1);
} // end composite()

inline ScatteringDistributionFunction *uber(const Spectrum &Kd,
                                            const Spectrum &Ks,
                                            const float uShininess,
                                            const float vShininess)
{
  return gContext->uber(Kd, Ks, uShininess, vShininess);
} // end uber()

inline ScatteringDistributionFunction *uber(const Spectrum &Kd,
                                            const Spectrum &Ks,
                                            const float shininess)
{
  return gContext->uber(Kd, Ks, shininess);
} // end uber()

inline ScatteringDistributionFunction *uber(const Spectrum &Ks,
                                            const float shininess,
                                            const Spectrum &Kr,
                                            const float eta)
{
  return gContext->uber(Ks, shininess, Kr, eta);
} // end uber()

inline ScatteringDistributionFunction *perspectiveSensor(const Spectrum &Ks,
                                                         const float aspect,
                                                         const Point &origin)
{
  return gContext->perspectiveSensor(Ks, aspect, origin);
} // end perspective()

inline ScatteringDistributionFunction *hemisphericalEmission(const Spectrum &Ke)
{
  return gContext->hemisphericalEmission(Ke);
} // end hemisphericalEmission()

