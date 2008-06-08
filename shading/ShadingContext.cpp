/*! \file ShadingContext.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ShadingContext class.
 */

#include "ShadingContext.h"

#include "PerspectiveSensor.h"
#include "HemisphericalEmission.h"
#include "Lambertian.h"
#include "PhongReflection.h"
#include "PhongTransmission.h"
#include "AshikhminShirleyReflection.h"
#include "SpecularReflection.h"
#include "SpecularTransmission.h"
#include "TransparentTransmission.h"
#include "CompositeDistributionFunction.h"
#include "PerfectGlass.h"
#include "ThinGlass.h"
#include "Fresnel.h"
#include "noise.h"

void ShadingContext
  ::setMaterials(const boost::shared_ptr<MaterialList> &materials)
{
  // keep our own list
  mMaterials.reset(new MaterialList());

  // copy the Materials
  *mMaterials = *materials;
} // end ShadingContext::setMaterials()

void ShadingContext
  ::setTextures(const boost::shared_ptr<TextureList> &textures)
{
  // keep our own list
  mTextures.reset(new TextureList());

  // copy the Textures
  *mTextures = *textures;
} // end ShadingContext::setTextures()

ShadingContext
  ::~ShadingContext(void)
{
  ;
} // end ShadingContext::~ShadingContext()

ScatteringDistributionFunction *ShadingContext
  ::diffuse(const Spectrum &Kd)
{
  return new(mAllocator) Lambertian(Kd);
} // end ShadingContext::diffuse()

ScatteringDistributionFunction *ShadingContext
  ::glossy(const Spectrum &Kr,
           const float eta,
           const float uExponent,
           const float vExponent)
{
  return new(mAllocator) AshikhminShirleyReflection(Kr, eta, uExponent, vExponent);
} // end ShadingContext::glossy()

ScatteringDistributionFunction *ShadingContext
  ::glossy(const Spectrum &Kr,
           const float eta,
           const float exponent)
{
  return new(mAllocator) PhongReflection(Kr, eta, exponent);
} // end ShadingContext::glossy()

ScatteringDistributionFunction *ShadingContext
  ::glossyRefraction(const Spectrum &Kt,
                     const float etai,
                     const float etat,
                     const float exponent)
{
  return new(mAllocator) PhongTransmission(Kt, etai, etat, exponent);
} // end ShadingContext::glossyRefraction()

ScatteringDistributionFunction *ShadingContext
  ::glass(const float eta,
          const Spectrum &Kr,
          const Spectrum &Kt)
{
  return new(mAllocator) PerfectGlass(Kr, Kt, 1.0f, eta);
} // end ShadingContext::glass()

ScatteringDistributionFunction *ShadingContext
  ::thinGlass(const float eta,
              const Spectrum &Kr,
              const Spectrum &Kt)
{
  return new(mAllocator) ThinGlass(Kr, Kt, 1.0f, eta);
} // end ShadingContext::thinGlass()

ScatteringDistributionFunction *ShadingContext
  ::mirror(const Spectrum &Kr,
           const float eta)
{
  return new(mAllocator) SpecularReflection(Kr, eta, mAllocator);
} // end ShadingContext::mirror()

ScatteringDistributionFunction *ShadingContext
  ::refraction(const Spectrum &Kt,
               const float etai,
               const float etat)
{
  return new(mAllocator) SpecularTransmission(Kt,etai,etat);
} // end ShadingContext::refraction()

ScatteringDistributionFunction *ShadingContext
  ::transparent(const Spectrum &Kt)
{
  return new(mAllocator) TransparentTransmission(Kt);
} // end ShadingContext::transparent()

ScatteringDistributionFunction *ShadingContext
  ::composite(ScatteringDistributionFunction *f0,
              ScatteringDistributionFunction *f1)
{
  CompositeDistributionFunction *c = new(mAllocator) CompositeDistributionFunction();
  *c += f0;
  *c += f1;
  return c;
} // end ShadingContext::composite()

ScatteringDistributionFunction *ShadingContext
  ::uber(const Spectrum &Kd,
         const Spectrum &Ks,
         const float uShininess,
         const float vShininess)
{
  ScatteringDistributionFunction *result = 0;

  ScatteringDistributionFunction *diffuse = 0;
  if(!Kd.isBlack())
  {
    diffuse = new(mAllocator) Lambertian(Kd);
  } // end if

  ScatteringDistributionFunction *specular = 0;
  if(!Ks.isBlack())
  {
    specular = glossy(Ks, Fresnel::approximateEta(Ks)[0], uShininess, vShininess);
  } // end if

  if(diffuse && specular)
  {
    CompositeDistributionFunction *c = new(mAllocator) CompositeDistributionFunction();
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
    result = new(mAllocator) ScatteringDistributionFunction();
  } // end else

  return result;
} // end ShadingContext::uber()

ScatteringDistributionFunction *ShadingContext
  ::uber(const Spectrum &Kd,
         const Spectrum &Ks,
         const float shininess)
{
  ScatteringDistributionFunction *result = 0;

  ScatteringDistributionFunction *diffuse = 0;
  if(!Kd.isBlack())
  {
    diffuse = new(mAllocator) Lambertian(Kd);
  } // end if

  ScatteringDistributionFunction *specular = 0;
  if(!Ks.isBlack())
  {
    if(shininess < 1000.0f)
    {
      specular = new(mAllocator) PhongReflection(Ks, Fresnel::approximateEta(Ks)[0], shininess);
    } // end if
    else
    {
      specular = new(mAllocator) SpecularReflection(Ks, Fresnel::approximateEta(Ks)[0], mAllocator);
    } // end else
  } // end if

  if(diffuse && specular)
  {
    CompositeDistributionFunction *c = new(mAllocator) CompositeDistributionFunction();
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
    result = new(mAllocator) ScatteringDistributionFunction();
  } // end else

  return result;
} // end ShadingContext::uber()

ScatteringDistributionFunction *ShadingContext
  ::uber(const Spectrum &Ks,
         const float shininess,
         const Spectrum &Kr,
         const float eta)
{
  ScatteringDistributionFunction *result = 0;

  ScatteringDistributionFunction *glossy = 0;
  if(!Ks.isBlack())
  {
    if(shininess < 1000.0f)
    {
      glossy = new(mAllocator) PhongReflection(Ks, eta, shininess);
    } // end if
    else
    {
      glossy = new(mAllocator) SpecularReflection(Ks, eta, mAllocator);
    } // end else
  } // end if

  ScatteringDistributionFunction *specular = 0;
  if(!Kr.isBlack())
  {
    specular = new(mAllocator) SpecularReflection(Ks, eta, mAllocator);
  } // end if

  if(glossy && specular)
  {
    CompositeDistributionFunction *c = new(mAllocator) CompositeDistributionFunction();
    *c += glossy;
    *c += specular;
    result = c;
  } // end if
  else if(glossy)
  {
    result = glossy;
  } // end else
  else if(specular)
  {
    result = specular;
  } // end else if
  else
  {
    // no scattering
    result = new(mAllocator) ScatteringDistributionFunction();
  } // end else

  return result;
} // end ShadingContext::uber()

ScatteringDistributionFunction *ShadingContext
  ::perspectiveSensor(const Spectrum &Ks,
                      const float aspect,
                      const Point &origin)
{
  return new(mAllocator) PerspectiveSensor(Ks, aspect, origin);
} // end ShadingContext::perspectiveSensor()

ScatteringDistributionFunction *ShadingContext
  ::hemisphericalEmission(const Spectrum &Ke)
{
  return new(mAllocator) HemisphericalEmission(Ke);
} // end ShadingContext::hemisphericalEmission()

ScatteringDistributionFunction *ShadingContext
  ::null(void)
{
  return new(mAllocator) ScatteringDistributionFunction();
} // end ShadingContext::null()

float ShadingContext
  ::noise(const float x, const float y, const float z)
{
  return ::noise(x, y, z);
} // end ShadingContext::noise()

Spectrum ShadingContext
  ::tex2D(const TextureHandle texture,
          const float u,
          const float v) const
{
  return getTexture(texture)->tex2D(u,v);
} // end ShadingContext::tex2D()

void ShadingContext
  ::freeAll(void)
{
  mAllocator.freeAll();
} // end ShadingContext::freeAll()

ScatteringDistributionFunction *ShadingContext
  ::clone(const ScatteringDistributionFunction *s)
{
  return s->clone(mAllocator);
} // end ShadingContext::clone()

FunctionAllocator &ShadingContext
  ::getAllocator(void)
{
  return mAllocator;
} // end ShadingContext::getAllocator()

ScatteringDistributionFunction *ShadingContext
  ::evaluateScattering(const MaterialHandle &m,
                       const DifferentialGeometry &dg)
{
  return (*mMaterials)[m]->evaluateScattering(*this, dg);
} // end ShadingContext::evaluateScattering()

void ShadingContext
  ::evaluateScattering(const MaterialHandle *m,
                       const DifferentialGeometry *dg,
                       const bool *stencil,
                       ScatteringDistributionFunction **f,
                       const size_t n)
{
  for(size_t i = 0; i != n; ++i)
  {
    if(stencil[i])
    {
      f[i] = evaluateScattering(m[i], dg[i]);
    } // end if
  } // end for i
} // end ShadingContext::evaluateScattering()

ScatteringDistributionFunction *ShadingContext
  ::evaluateSensor(const MaterialHandle &m,
                   const DifferentialGeometry &dg)
{
  return (*mMaterials)[m]->evaluateSensor(*this, dg);
} // end ShadingContext::evaluateSensor()

void ShadingContext
  ::evaluateSensor(const MaterialHandle *m,
                   const DifferentialGeometry *dg,
                   const bool *stencil,
                   ScatteringDistributionFunction **f,
                   const size_t n)
{
  for(size_t i = 0; i != n; ++i)
  {
    if(stencil[i])
    {
      f[i] = evaluateSensor(m[i], dg[i]);
    } // end if
  } // end for i
} // end ShadingContext::evaluateSensor()

ScatteringDistributionFunction *ShadingContext
  ::evaluateEmission(const MaterialHandle &m,
                     const DifferentialGeometry &dg)
{
  return (*mMaterials)[m]->evaluateEmission(*this, dg);
} // end ShadingContext::evaluateEmission()

void ShadingContext
  ::evaluateEmission(const MaterialHandle *m,
                     const DifferentialGeometry *dg,
                     const bool *stencil,
                     ScatteringDistributionFunction **f,
                     const size_t n)
{
  for(size_t i = 0; i != n; ++i)
  {
    if(stencil[i])
    {
      f[i] = evaluateEmission(m[i], dg[i]);
    } // end if
  } // end for i
} // end ShadingContext::evaluateEmission()

void ShadingContext
  ::evaluateBidirectionalScattering(ScatteringDistributionFunction **f,
                                    const Vector *wo,
                                    const DifferentialGeometry *dg,
                                    const Vector *wi,
                                    const bool *stencil,
                                    Spectrum *results,
                                    const size_t n)
{
  for(size_t i = 0; i != n; ++i)
  {
    if(stencil[i])
    {
      results[i] = f[i]->evaluate(wo[i], dg[i], wi[i]);
    } // end if
  } // end for i
} // end ShadingContext::evaluateBidirectionalScattering()

void ShadingContext
  ::evaluateUnidirectionalScattering(ScatteringDistributionFunction **f,
                                     const Vector *wo,
                                     const DifferentialGeometry *dg,
                                     const bool *stencil,
                                     Spectrum *results,
                                     const size_t n)
{
  for(size_t i = 0; i != n; ++i)
  {
    if(stencil[i])
    {
      results[i] = f[i]->evaluate(wo[i], dg[i]);
    } // end if
  } // end for i
} // end ShadingContext::evaluateUnidirectionalScattering()

const Material *ShadingContext
  ::getMaterial(const MaterialHandle &h) const
{
  return (*mMaterials)[h].get();
} // end ShadingContext::getMaterial()

const Texture *ShadingContext
  ::getTexture(const TextureHandle &h) const
{
  return (*mTextures)[h].get();
} // end ShadingContext::getTexture()

void ShadingContext
  ::postprocess(void)
{
  ;
} // end ShadingContext::postprocess()

