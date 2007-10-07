/*! \file noises.h
 *  \author Jared Hoberock
 *  \brief Adapted from Gelato/shaders/noises.h.
 */

#ifndef NOISES_H
#define NOISES_H

#include "noise.h"
#include "../geometry/Point.h"

inline float snoise(const Point &x)
{
  return gotham::noise(x[0], x[1], x[2]);
} // end snoise()

inline float snoise(float x, float y, float z)
{
  return gotham::noise(x,y,z);
} // end snoise()

inline float snoise(const float x)
{
  return gotham::noise(x,0,0);
} // end snoise()

inline float periodicFbm(float p, float period, int octaves, float gain)
{
  float amp = 1.0f;
  float pp = p;
  float pper = period;
  float sum = 0;

  for(int i = 0; i < octaves; ++i)
  {
    sum += amp * snoise(pp);
    amp *= gain;
    pp *= 2.0f;
    pper *= 2.0f;
  } // end for i

  return sum;
} // end periodicFbm()

#endif // NOISES_H

