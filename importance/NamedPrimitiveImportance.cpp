/*! \file NamedPrimitiveImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of NamedPrimitiveImportance class.
 */

#include "NamedPrimitiveImportance.h"
#include "MaxImportance.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "../primitives/SurfacePrimitive.h"

NamedPrimitiveImportance
  ::NamedPrimitiveImportance(const std::string &name,
                             const float factor)
    :Parent(),
     mName(name),
     mNameHash(mHasher(name)),
     mFactor(factor)
{
  ;
} // end NamedPrimitiveImportance::NamedPrimitiveImportance()

float NamedPrimitiveImportance
  ::evaluate(const PathSampler::HyperPoint &x,
             const Path &xPath,
             const std::vector<PathSampler::Result> &results)
{
  float I = 0;
  for(std::vector<PathSampler::Result>::const_iterator r = results.begin();
      r != results.end();
      ++r)
  {
    I += MaxImportance::evaluateImportance(x,xPath,*r);
  } // end for r

  // find the first PathVertex after the sensor which isn't specular
  // check if it lies on the Primitive of interest
  float scale = 1.0f;
  for(size_t i = 1;
      i != xPath.getSubpathLengths().sum();
      ++i)
  {
    if(!xPath[i].mScattering->isSpecular())
    {
      // found the first non-specular surface
      if(xPath[i].mSurface->getNameHash() == mNameHash)
      {
        // found the primitive we were looking for
        scale = mFactor;
      } // end if

      break;
    } // end if
  } // end for i

  return scale * I;
} // end NamedPrimitiveImportance::evaluate()

