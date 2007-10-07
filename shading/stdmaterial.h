/*! \file stdmaterial.h
 *  \author Jared Hoberock
 *  \brief Defines some functions for creating
 *         ScatteringReflectionFunctions for commonly used
 *         materials.
 */

#include "SpecularReflection.h"
#include "SpecularTransmission.h"
#include "CompositeDistributionFunction.h"
#include "PerfectGlass.h"
#include "ThinGlass.h"
#include "Fresnel.h"

inline ScatteringDistributionFunction *glass(const float eta,
                                             const Spectrum &Kr,
                                             const Spectrum &Kt)
{
  return new PerfectGlass(Kr, Kt, 1.0f, eta);
} // end glass()

inline ScatteringDistributionFunction *thinGlass(const float eta,
                                                 const Spectrum &Kr,
                                                 const Spectrum &Kt)
{
  return new ThinGlass(Kr, Kt, 1.0f, eta);
} // end thinGlass()

inline ScatteringDistributionFunction *uber(const Spectrum &Kd,
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
    specular = new AshikhminShirleyReflection(Ks, Fresnel::approximateEta(Ks)[0], uShininess, vShininess);
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
} // end uber()

inline ScatteringDistributionFunction *uber(const Spectrum &Kd,
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
} // end uber()

