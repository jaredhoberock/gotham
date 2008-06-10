/*! \file stdshader.h
 *  \brief Adapted from Gelato/shaders/stdshader.h
 */

#ifndef STDSHADER_H
#define STDSHADER_H

#include "exportShading.h"

#include "Spectrum.h"
#include "Point.h"
#include "Vector.h"

#include "stdmaterial.h"
#include "noises.h"

#ifndef PI
#define PI 3.14159265f
#endif // PI

template<typename T>
  inline T clamp(const T &v, const T &m, const T &M)
{
  return v < m ? m : ((v > M) ? M : v);
} // end clamp()

template<typename T>
  inline T mix(const T &x, const T &y, float alpha)
{
  return x * (1.0f - alpha) + y * alpha;
} // end mix()

template<typename T>
  inline T lerp(const T &t, const T &a, const T &b)
{
  return a + t * (b - a);
} // end lerp()

template<typename T>
  inline T fade(const T &t)
{
  return t * t * t * (t * (t * static_cast<T>(6) - static_cast<T>(15)) + static_cast<T>(10));
} // end fade()

template<typename T>
  inline T smoothstep(const T a, const T b, const T t)
{
  if(t < a) return 0;
  else if(t > b) return 1;
  float x = (t - a) / (b - a);
  return x*x*(static_cast<T>(3) - static_cast<T>(2)*x);
} // end smoothstep()

// The idea here is that we need a context to evaluate
// some shading functions
// So we will require that the shader set this at the beginning of its
// function before calling any of these functions.
// XXX This is probably not reentrant
extern ShadingInterface *gContext;

inline Spectrum tex2D(const TextureHandle texture,
                      const float u,
                      const float v)
{
  return gContext->tex2D(texture, u, v);
} // end tex2D()

#include "patterns.h"

#endif // STDSHADER_H

