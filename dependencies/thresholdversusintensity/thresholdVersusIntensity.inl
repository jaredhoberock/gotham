/*! \file thresholdVersusIntensity.inl
 *  \author Jared Hoberock
 *  \brief Inline file for thresholdVersusIntensity.h.
 */

#include "thresholdVersusIntensity.h"
#include <math.h>

float tvi(const float Y)
{
  float logY = log10f(std::max<float>(0.0f,Y));
  float logResult = 0;

  if(logY < -3.94f)
  {
    logResult = -2.86f;
  } // end if
  else if(logY < -1.44f)
  {
    logResult = powf(0.405f * logY + 1.6f, 2.18f) - 2.86f;
  } // end else if
  else if(logY < -0.0184f)
  {
    logResult = logY - 0.395f;
  } // end else
  else if(logY < 1.9f)
  {
    logResult = powf(0.249f * logY + 0.65f, 2.7f) - 0.72f;
  } // end else if
  else
  {
    logResult = logY - 1.255f;
  } // end else

  return powf(10.0f, logResult);
} // end tvi()

