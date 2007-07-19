/*! \file FunctionAllocator.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of FunctionAllocator class.
 */

#include "FunctionAllocator.h"
#include "ScatteringDistributionFunction.h"
#include "Lambertian.h"
#include "HemisphericalEmission.h"
#include "PerspectiveSensor.h"
#include <vector>

template<unsigned int size>
  struct Accomodator
{
  unsigned char mFill[size];
}; // end Accomodator

// pre-allocate a huge number of slots
std::vector<Accomodator<sizeof(PerspectiveSensor)> > gStorage;

void *FunctionAllocator
  ::malloc(void)
{
  // XXX don't call this each time
  gStorage.reserve(32678);

  if(gStorage.size() < gStorage.capacity())
  {
    gStorage.resize(gStorage.size() + 1);
    return &gStorage.back();
  } // end if

  return 0;
} // end FunctionAllocator::malloc()

void FunctionAllocator
  ::freeAll(void)
{
  gStorage.clear();
} // end FunctionAllocator::freeAll()

