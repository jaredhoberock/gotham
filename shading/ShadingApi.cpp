/*! \file ShadingApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ShadingApi class.
 */

#include "ShadingApi.h"

using namespace boost;

ShadingContext *ShadingApi
  ::context(Gotham::AttributeMap &attr,
            const boost::shared_ptr<MaterialList> &materials)
{
  ShadingContext *result = 0;

  result = new ShadingContext();

  // give the materials to the ShadingContext
  result->setMaterials(materials);

  return result;
} // end context()

