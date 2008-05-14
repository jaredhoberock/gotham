/*! \file noises.h
 *  \author Jared Hoberock
 *  \brief Adapted from Gelato/shaders/noises.h.
 */

#ifndef NOISES_H
#define NOISES_H

#include "stdshader.h"
#include "ShadingInterface.h"
#include "Point.h"

// A hook into the ShadingInterface
// XXX This is probably not reentrant
extern ShadingInterface *gContext;

inline float snoise(const Point &x)
{
  return gContext->noise(x[0], x[1], x[2]);
} // end snoise()

inline float snoise(float x, float y, float z)
{
  return gContext->noise(x,y,z);
} // end snoise()

inline float snoise(const float x)
{
  return gContext->noise(x,0,0);
} // end snoise()

inline Vector vsnoise(const Point &x)
{
  // XXX This is almost definitely not robust
  return Vector(snoise(x),
                snoise(x + Point(1,1,1)),
                snoise(x + Point(2,2,2)));
} // end vsnoise()

#ifndef FILTLOW
#  define FILTLOW 0.2
#endif
#ifndef FILTHIGH
#  define FILTHIGH 0.75
#endif

inline Vector vectorFbm(const Point &p,
                        int octaves, float lacunarity, float gain)
{
  float amp = 1.0f;
  Point pp = p;
  Vector sum(0,0,0);

  for(int i = 0; i < octaves; ++i)
  {
    sum += amp * vsnoise(pp);
    amp *= gain;

    // multiply and offset
    pp = pp * lacunarity + Point(1,1,1);
  } // end for i

  return sum;
} // end vectorFbm()

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

