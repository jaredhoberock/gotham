/*! \file FunctionAllocator.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of FunctionAllocator class.
 */

#include "FunctionAllocator.h"
#include "ScatteringDistributionFunction.h"
#include "CompositeDistributionFunction.h"
#include "Lambertian.h"
#include "HemisphericalEmission.h"
#include "PerspectiveSensor.h"
#include "SpecularReflection.h"
#include "SpecularTransmission.h"
#include "PerfectGlass.h"
#include "ThinGlass.h"
#include "TransparentTransmission.h"
#include "PhongReflection.h"
#include "PhongTransmission.h"
#include "AshikhminShirleyReflection.h"
#include "Fresnel.h"
#include <vector>
#include <boost/static_assert.hpp>
#include <assert.h>

FunctionAllocator
  ::FunctionAllocator(void)
{
  typedef Accomodator<8 * sizeof(size_t)> Block;

  // XXX TODO get these lines to compile on 64b (they should all assert true)
  //// assert that any known ScatteringDistributionFunction will fit into a Block
  //BOOST_STATIC_ASSERT(sizeof(Block) >= sizeof(PerspectiveSensor));
  //BOOST_STATIC_ASSERT(sizeof(Block) >= sizeof(Lambertian));
  //BOOST_STATIC_ASSERT(sizeof(Block) >= sizeof(HemisphericalEmission));
  //BOOST_STATIC_ASSERT(sizeof(Block) >= sizeof(SpecularReflection));
  //BOOST_STATIC_ASSERT(sizeof(Block) >= sizeof(SpecularTransmission));
  //BOOST_STATIC_ASSERT(sizeof(Block) >= sizeof(PhongReflection));
  //BOOST_STATIC_ASSERT(sizeof(Block) >= sizeof(PhongTransmission));
  //BOOST_STATIC_ASSERT(sizeof(Block) >= sizeof(CompositeDistributionFunction));
  //BOOST_STATIC_ASSERT(sizeof(Block) >= sizeof(TransparentTransmission));
  //BOOST_STATIC_ASSERT(sizeof(Block) >= sizeof(PerfectGlass));
  //BOOST_STATIC_ASSERT(sizeof(Block) >= sizeof(ThinGlass));
  //BOOST_STATIC_ASSERT(sizeof(Block) >= sizeof(AshikhminShirleyReflection));

  //// assert that either Fresnel will fit into a block
  //BOOST_STATIC_ASSERT(sizeof(Block) >= sizeof(FresnelDielectric));
  //BOOST_STATIC_ASSERT(sizeof(Block) >= sizeof(FresnelConductor));

  reserve(32678);
} // end FunctionAllocator::FunctionAllocator()

FunctionAllocator
  ::FunctionAllocator(const size_t n)
{
  reserve(n);
} // end FunctionAllocator::FunctionAllocator()

void FunctionAllocator
  ::reserve(const size_t n)
{
  mStorage.reserve(n);
} // end FunctionAllocator::reserve()

void *FunctionAllocator
  ::malloc(void)
{
  if(mStorage.size() < mStorage.capacity())
  {
    mStorage.resize(mStorage.size() + 1);
    return &mStorage.back();
  } // end if

  return 0;
} // end FunctionAllocator::malloc()

void FunctionAllocator
  ::freeAll(void)
{
  mStorage.clear();
} // end FunctionAllocator::freeAll()

