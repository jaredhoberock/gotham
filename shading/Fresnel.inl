/*! \file Fresnel.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Fresnel.h.
 */

#include "Fresnel.h"

Fresnel
  ::~Fresnel(void)
{
  ;
} // end Fresnel::~Fresnel()

float Fresnel
  ::dielectric(const float etai, const float etat,
               const float cosi, const float cost)
{
  float rParallel = ((etat * cosi) - (etai * cost)) /
                    ((etat * cosi) + (etai * cost));

  float rPerpendicular = ((etai * cosi) - (etat * cost)) /
                         ((etai * cosi) + (etat * cost));
  return std::min(1.0f, (rParallel * rParallel + rPerpendicular * rPerpendicular) / 2.0f);
} // end Fresnel::dielectric()

float Fresnel
  ::dielectric(const float etai, const float etat,
               float cosi)
{
  // clamp cosi
  cosi = std::min(cosi, 1.0f);
  cosi = std::max(cosi, -1.0f);

  // compute indices of refraction for dielectric
  bool entering = cosi > 0;
  float ei = etai, et = etat;
  if(!entering) std::swap(ei,et);

  // compute sint
  float sint = ei/et * sqrt(std::max(0.0f, 1.0f - cosi*cosi));

  // handle total internal reflection
  if(sint > 1.0f) return 1.0f;

  float cost = sqrt(std::max(0.0f, 1.0f - sint*sint));
  return dielectric(ei, et, cosi, cost);
} // end Fresnel::dielectric()

Spectrum Fresnel
  ::conductor(const float cosi, const Spectrum &eta,
              const Spectrum &k)
{
  float c2 = cosi*cosi;
  Spectrum cosi2(c2, c2, c2);
  Spectrum tmp = cosi2 * (eta * eta + k * k);

  Spectrum twoEtaCosi = 2.0f * cosi * eta;

  Spectrum one(1,1,1);

  Spectrum rParallel2 = (tmp - twoEtaCosi + one) /
                        (tmp + twoEtaCosi + one);
  Spectrum tmpf = eta * eta + k * k;

  Spectrum rPerp2 = (tmpf - twoEtaCosi + cosi2) /
                    (tmpf + twoEtaCosi + cosi2);

  return (rParallel2 + rPerp2) / 2.0f;
} // end Fresnel::conductor()

Spectrum Fresnel
  ::approximateEta(const Spectrum &r)
{
  Spectrum root = r;
  root[0] = std::min(std::max(root[0], 0.99999f), 0.0f);
  root[1] = std::min(std::max(root[1], 0.99999f), 0.0f);
  root[2] = std::min(std::max(root[2], 0.99999f), 0.0f);

  root[0] = sqrt(root[0]);
  root[1] = sqrt(root[1]);
  root[2] = sqrt(root[2]);

  Spectrum one(1,1,1);

  return (one + root) / (one - root);
} // end Fresnel::approximateEta()

Spectrum Fresnel
  ::approximateAbsorbance(const Spectrum &r)
{
  Spectrum refl = r;
  refl[0] = std::min(std::max(refl[0], 0.99999f), 0.0f);
  refl[1] = std::min(std::max(refl[1], 0.99999f), 0.0f);
  refl[2] = std::min(std::max(refl[2], 0.99999f), 0.0f);

  refl /= (Spectrum(1,1,1) - refl);

  refl[0] = sqrt(refl[0]);
  refl[1] = sqrt(refl[1]);
  refl[2] = sqrt(refl[2]);

  return 2.0f * refl;
} // end Fresnel::approximateAbsorbance()

