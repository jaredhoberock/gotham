/*! \file bittricks.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to several bit-twiddling tricks.
 */

#ifndef BITTRICKS_H
#define BITTRICKS_H

/*! This function computes the integer floor of
 *  a float faster than a cast.
 *  \param t The decimal of interest.
 *  \return The integer floor of t.
 */
inline int ifloor(const double t);

/*! This function rounds a decimal to an
 *  integer faster than a cast.
 *  \param t The decimal of interest.
 *  \return The nearest integer to t.
 */
inline int iround(const double t);

/*! This function interprets a word for another word.
 *  \param word The word to interpret.
 *  \note assert(sizeof(IN) == sizeof(OUT))
 *  */
template<typename OUT_TYPE, typename IN_TYPE>
  inline OUT_TYPE reinterpretWord(const IN_TYPE &word);

#include "bittricks.inl"

inline float        sizeAsFloat(const size_t &sz) {return reinterpretWord<float,size_t>(sz);}
inline size_t       floatAsSize(const float &f) {return reinterpretWord<size_t,float>(f);}
inline float        intAsFloat(const int &i) {return reinterpretWord<float,int>(i);}
inline int          floatAsInt(const float &f) {return reinterpretWord<int,float>(f);}
inline float        uintAsFloat(const unsigned int &ui) {return reinterpretWord<float,unsigned int>(ui);}
inline unsigned int floatAsUint(const float &f) {return reinterpretWord<unsigned int,float>(f);}

#endif // BITTRICKS_H

