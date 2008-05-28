/*! \file ShadingApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ShadingApi class.
 */

#include "ShadingApi.h"

using namespace boost;

ShadingContext *ShadingApi
  ::context(Gotham::AttributeMap &attr,
            const boost::shared_ptr<MaterialList> &materials,
            const boost::shared_ptr<TextureList> &textures)
{
  ShadingContext *result = 0;

  result = new ShadingContext();

  // give the materials to the ShadingContext
  result->setMaterials(materials);

  // give the textures to the ShadingContext
  result->setTextures(textures);

  return result;
} // end context()

