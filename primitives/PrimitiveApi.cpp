/*! \file PrimitiveApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PrimitiveApi class.
 */

#include "PrimitiveApi.h"
#include "../rasterizables/RasterizableScene.h"
#include "../rasterizables/RasterizablePrimitiveList.h"
#include "../primitives/PrimitiveBSP.h"
#include "../surfaces/Mesh.h"
#include "TriangleBVH.h"
#include "Scene.h"
#include "UnshadowedScene.h"
#include <algorithm>

using namespace boost;

PrimitiveList *PrimitiveApi
  ::list(Gotham::AttributeMap &attr,
         const PrimitiveList &prims)
{
  PrimitiveList *result = 0;

  bool allMeshes = true;

  // if everything is a mesh, use a TriangleBVH
  for(PrimitiveList::const_iterator prim = prims.begin();
      prim != prims.end();
      ++prim)
  {
    const SurfacePrimitive *sp = dynamic_cast<const SurfacePrimitive*>(prim->get());
    if(sp)
    {
      const Mesh *mesh = dynamic_cast<const Mesh *>(sp->getSurface());
      if(!mesh)
      {
        allMeshes = false;
        break;
      } // end else
    } // end if
    else
    {
      allMeshes = false;
      break;
    } // end else
  } // end for prim

  if(allMeshes)
  {
    result = new RasterizablePrimitiveList<TriangleBVH>();
  } // end if
  else
  {
    result = new RasterizablePrimitiveList<PrimitiveBSP>();
  } // end else

  // copy the prims
  std::copy(prims.begin(), prims.end(), std::back_inserter(*result));

  return result;
} // end PrimitiveApi::list()

SurfacePrimitiveList *PrimitiveApi
  ::surfacesList(Gotham::AttributeMap &attr,
                 const SurfacePrimitiveList &surfaces)
{
  SurfacePrimitiveList *result = 0;

  result = new RasterizablePrimitiveList<SurfacePrimitiveList>();

  // copy the prims
  std::copy(surfaces.begin(), surfaces.end(), std::back_inserter(*result));

  return result;
} // end PrimitiveApi::surfacesList()

Scene *PrimitiveApi
  ::scene(Gotham::AttributeMap &attr)
{
  Scene *result = 0;

  if(attr["scene:castshadows"] == std::string("false"))
  {
    result = new RasterizableScene<UnshadowedScene>();
  } // end if
  else
  {
    result = new RasterizableScene<Scene>();
  } // end else

  return result;
} // end PrimitiveApi::scene()

