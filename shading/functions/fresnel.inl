/*! \file fresnel.inl
 *  \author Jared Hoberock
 *  \brief Inline file for fresnel.h.
 */

#include "fresnel.h"

float evaluateFresnelDielectric(const float ei,
                                const float et,
                                const float cosi,
                                const float cost)
{
  float rParallel = ((et * cosi) - (ei * cost)) /
                    ((et * cosi) + (ei * cost));

  float rPerpendicular = ((ei * cosi) - (et * cost)) /
                         ((ei * cosi) + (et * cost));

  float result = (rParallel * rParallel + rPerpendicular * rPerpendicular) / 2.0f;

  // clamp to 1
  result = result > 1.0f ? 1.0f : result;

  return result;
} // end evaluateFresnelDielectric()

