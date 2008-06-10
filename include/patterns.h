/*! \file patterns.h
 *  \author Jared Hoberock
 *  \brief Adapted from Gelato/shaders/patterns.h.
 */

#ifndef PATTERNS_H
#define PATTERNS_H

template<typename T>
  inline T step(T edge, T x)
{
  return x < edge ? 0 : 1;
} // end step()

// A 1-D pulse pattern:  return 1 if edge0 <= x <= edge1, otherwise 0
template<typename T>
  inline T pulse(T edge0, T edge1, T x)
{
  return step(edge0,x) - step(edge1,x);
} // end pulse()

template<typename T>
  inline T pulsetrain(T edge, T period, T x)
{
  return pulse(edge, period, fmodf(x,period));
} // end pulsetrain()

inline float smoothpulse(float e0, float e1, float e2, float e3, float x)
{
  return smoothstep(e0,e1,x) - smoothstep(e2,e3,x);
} // end smoothpulse

// A pulse train of smoothsteps: a signal that repeats with a given
// period, and is 0 when 0 <= mod(x/period,1) < edge, and 1 when
// mod(x/period,1) > edge.  
//
inline float smoothpulsetrain(float e0, float e1, float e2, float e3,
                              float period, float x)
{
  return smoothpulse(e0, e1, e2, e3, fmod(x,period));
} // end smoothpulsetrain()

inline float checker(const float s, const float t,
                     const float sfreq, const float tfreq)
{
  float x = pulsetrain(1.0f, 2.0f, s * sfreq);
  float y = pulsetrain(1.0f, 2.0f, t * tfreq);
  return x*y  + (1.0f - x) * (1.0f - y);
} // end checker()

#endif // PATTERNS_H

