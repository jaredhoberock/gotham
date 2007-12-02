/*! \file ShaderApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ShaderApi class.
 */

#include "ShaderApi.h"

#include "../shading/PerspectiveSensor.h"
#include "../shading/HemisphericalEmission.h"
#include "../shading/Lambertian.h"
#include "../shading/PhongReflection.h"
#include "../shading/PhongTransmission.h"
#include "../shading/AshikhminShirleyReflection.h"
#include "../shading/SpecularReflection.h"
#include "../shading/SpecularTransmission.h"
#include "../shading/TransparentTransmission.h"
#include "../shading/CompositeDistributionFunction.h"
#include "../shading/PerfectGlass.h"
#include "../shading/ThinGlass.h"
#include "../shading/Fresnel.h"

ScatteringDistributionFunction *ShaderApi
  ::diffuse(const Spectrum &Kd)
{
  return new Lambertian(Kd);
} // end ShaderApi::diffuse()

ScatteringDistributionFunction *ShaderApi
  ::glossy(const Spectrum &Kr,
           const float eta,
           const float uExponent,
           const float vExponent)
{
  return new AshikhminShirleyReflection(Kr, eta, uExponent, vExponent);
} // end ShaderApi::glossy()

ScatteringDistributionFunction *ShaderApi
  ::glossy(const Spectrum &Kr,
           const float eta,
           const float exponent)
{
  return new PhongReflection(Kr, eta, exponent);
} // end ShaderApi::glossy()

ScatteringDistributionFunction *ShaderApi
  ::glossyRefraction(const Spectrum &Kt,
                     const float etai,
                     const float etat,
                     const float exponent)
{
  return new PhongTransmission(Kt, etai, etat, exponent);
} // end ShaderApi::glossyRefraction()

ScatteringDistributionFunction *ShaderApi
  ::glass(const float eta,
          const Spectrum &Kr,
          const Spectrum &Kt)
{
  return new PerfectGlass(Kr, Kt, 1.0f, eta);
} // end ShaderApi::glass()

ScatteringDistributionFunction *ShaderApi
  ::thinGlass(const float eta,
              const Spectrum &Kr,
              const Spectrum &Kt)
{
  return new ThinGlass(Kr, Kt, 1.0f, eta);
} // end ShaderApi::thinGlass()

ScatteringDistributionFunction *ShaderApi
  ::mirror(const Spectrum &Kr,
           const float eta)
{
  return new SpecularReflection(Kr, eta);
} // end ShaderApi::mirror()

ScatteringDistributionFunction *ShaderApi
  ::refraction(const Spectrum &Kt,
               const float etai,
               const float etat)
{
  return new SpecularTransmission(Kt,etai,etat);
} // end ShaderApi::refraction()

ScatteringDistributionFunction *ShaderApi
  ::transparent(const Spectrum &Kt)
{
  return new TransparentTransmission(Kt);
} // end ShaderApi::transparent()

ScatteringDistributionFunction *ShaderApi
  ::uber(const Spectrum &Kd,
         const Spectrum &Ks,
         const float uShininess,
         const float vShininess)
{
  ScatteringDistributionFunction *result = 0;

  ScatteringDistributionFunction *diffuse = 0;
  if(!Kd.isBlack())
  {
    diffuse = new Lambertian(Kd);
  } // end if

  ScatteringDistributionFunction *specular = 0;
  if(!Ks.isBlack())
  {
    specular = glossy(Ks, Fresnel::approximateEta(Ks)[0], uShininess, vShininess);
  } // end if

  if(diffuse && specular)
  {
    CompositeDistributionFunction *c = new CompositeDistributionFunction();
    *c += diffuse;
    *c += specular;
    result = c;
  } // end if
  else if(diffuse)
  {
    result = diffuse;
  } // end else
  else if(specular)
  {
    result = specular;
  } // end else if
  else
  {
    // no scattering
    result = new ScatteringDistributionFunction();
  } // end else

  return result;
} // end ShaderApi::uber()

ScatteringDistributionFunction *ShaderApi
  ::uber(const Spectrum &Kd,
         const Spectrum &Ks,
         const float shininess)
{
  ScatteringDistributionFunction *result = 0;

  ScatteringDistributionFunction *diffuse = 0;
  if(!Kd.isBlack())
  {
    diffuse = new Lambertian(Kd);
  } // end if

  ScatteringDistributionFunction *specular = 0;
  if(!Ks.isBlack())
  {
    if(shininess < 1000.0f)
    {
      specular = new PhongReflection(Ks, Fresnel::approximateEta(Ks)[0], shininess);
    } // end if
    else
    {
      specular = new SpecularReflection(Ks, Fresnel::approximateEta(Ks)[0]);
    } // end else
  } // end if

  if(diffuse && specular)
  {
    CompositeDistributionFunction *c = new CompositeDistributionFunction();
    *c += diffuse;
    *c += specular;
    result = c;
  } // end if
  else if(diffuse)
  {
    result = diffuse;
  } // end else
  else if(specular)
  {
    result = specular;
  } // end else if
  else
  {
    // no scattering
    result = new ScatteringDistributionFunction();
  } // end else

  return result;
} // end ShaderApi::uber()

ScatteringDistributionFunction *ShaderApi
  ::perspectiveSensor(const Spectrum &Ks,
                      const float aspect,
                      const Point &origin,
                      const Vector &right,
                      const Vector &up)
{
  return new PerspectiveSensor(Ks, aspect, origin, right, up);
} // end ShaderApi::perspectiveSensor()

ScatteringDistributionFunction *ShaderApi
  ::hemisphericalEmission(const Spectrum &Ke)
{
  return new HemisphericalEmission(Ke);
} // end ShaderApi::hemisphericalEmission()

ScatteringDistributionFunction *ShaderApi
  ::null(void)
{
  return new ScatteringDistributionFunction();
} // end ShaderApi::null()

