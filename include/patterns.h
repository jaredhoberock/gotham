/*! \file patterns.h
 *  \author Jared Hoberock
 *  \brief Adapted from Gelato/shadres/patterns.h.
 */

#ifndef PATTERNS_H
#define PATTERNS_H

template<typename T>
  T step(T edge, T x)
{
  return x < edge ? 0 : 1;
} // end step()

// A 1-D pulse pattern:  return 1 if edge0 <= x <= edge1, otherwise 0
template<typename T>
  T pulse(T edge0, T edge1, T x)
{
  return step(edge0,x) - step(edge1,x);
} // end pulse()

template<typename T>
  T pulsetrain(T edge, T period, T x)
{
  return pulse(edge, period, fmodf(x,period));
} // end pulsetrain()

#endif // PATTERNS_H

