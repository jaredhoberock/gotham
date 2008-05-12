/*! \file CudaShadingApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaShadingAPi class.
 */

#include "CudaShadingApi.h"
#include "ScatteredAccessContext.h"

using namespace boost;

CudaShadingContext *CudaShadingApi
  ::context(Gotham::AttributeMap &attr,
            const boost::shared_ptr<MaterialList> &materials)
{
  CudaShadingContext *result = 0;

  result = new ScatteredAccessContext();

  // give the materials to the CudaShadingContext
  result->setMaterials(materials);

  return result;
} // end context()

