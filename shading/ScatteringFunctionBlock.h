/*! \file ScatteringFunctionBlock.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a chunk of
 *         memory large enough to hold any one of
 *         our ScatteringDistributionFunctions.
 */

#pragma once

// XXX this is nasty: kill it
#ifndef WIN32
#define DLLAPI
#endif // WIN32

template<unsigned int size>
  struct DLLAPI Accomodator
{
  unsigned char mFill[size];
}; // end Accomodator

typedef Accomodator<18 * sizeof(int)> ScatteringFunctionBlock;

