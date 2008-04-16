/*! \file ShadingApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ShadingApi class.
 */

#include "ShadingApi.h"
#include "../cuda/shading/ScatteredAccessContext.h"

using namespace boost;

ShadingContext *ShadingApi
  ::context(Gotham::AttributeMap &attr,
            const boost::shared_ptr<MaterialList> &materials)
{
  ShadingContext *result = 0;

  size_t numThreads = lexical_cast<size_t>(attr["renderer:threads"]);
  if(numThreads > 1)
  {
    result = new ScatteredAccessContext();
  } // end if
  else
  {
    result = new ShadingContext();
  } // end else

  // give the materials to the ShadingContext
  result->setMaterials(materials);

  return result;
} // end context()

