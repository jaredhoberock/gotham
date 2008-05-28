/*! \file bittricks.inl
 *  \author Jared Hoberock
 *  \brief Inline file for bittricks.inl.
 */

#include "bittricks.h"

#define _doublemagicroundeps	      (.5-1.4e-11)
//almost .5f = .5f - 1e^(number of exp bit)
 
int iround(double t)
{
  static const double _doublemagic = 6755399441055744.0;
  //2^52 * 1.5,  uses limited precision to floor
  t	= t + _doublemagic;
  // XXX hack around reinterpret_cast not working correctly on gcc 4.1
  //     \see http://gcc.gnu.org/ml/gcc-help/2006-06/msg00100.html
  //     This is also similar to CUDA's implementation of __float_as_int()
  //return (reinterpret_cast<long*>(&t))[0];
  union {double d; long l;} u;
  u.d = t;
  return u.l;
} // end iround()

int ifloor(double t)
{
  // XXX BUG this doesn't seem to be working
  //return iround(t - _doublemagicroundeps);
  return static_cast<int>(floorf(t));
} // end ifloor()

template<typename OUT, typename IN>
  OUT reinterpretWord(const IN &word)
{
  // XXX hack around reinterpret_cast not working correctly on gcc 4.1
  //     \see http://gcc.gnu.org/ml/gcc-help/2006-06/msg00100.html
  //     This is also similar to CUDA's implementation of __float_as_int()
  union {IN i; OUT o;} u;
  u.i = word;
  return u.o;
} // end reinterpretWord()

