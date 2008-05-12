/*! \file PrimitiveApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PrimitiveApi class.
 */

#include "PrimitiveApi.h"
#include "../rasterizables/RasterizableScene.h"
#include "../rasterizables/RasterizablePrimitiveList.h"
#include "../primitives/PrimitiveBSP.h"
#include "TriangleBVH.h"
#include "../cuda/primitives/CudaTriangleBVH.h"
#include "../cuda/primitives/CudaScene.h"
#include "Scene.h"
#include "UnshadowedScene.h"
#include <algorithm>

using namespace boost;

PrimitiveList *PrimitiveApi
  ::list(Gotham::AttributeMap &attr,
         const PrimitiveList &prims)
{
  PrimitiveList *result = 0;

  size_t numThreads = lexical_cast<size_t>(attr["renderer:threads"]);

  if(numThreads > 1)
  {
    CudaTriangleBVH *bvh = new RasterizablePrimitiveList<CudaTriangleBVH>();
    bvh->setWorkBatchSize(numThreads);
    result = bvh;
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

  size_t numThreads = lexical_cast<size_t>(attr["renderer:threads"]);

  if(numThreads > 1)
  {
    // XXX no need for this to be a bvh
    CudaTriangleBVH *bvh = new RasterizablePrimitiveList<CudaTriangleBVH>();
    bvh->setWorkBatchSize(numThreads);
    result = bvh;
  } // end if
  else
  {
    result = new RasterizablePrimitiveList<SurfacePrimitiveList>();
  } // end else

  // copy the prims
  std::copy(surfaces.begin(), surfaces.end(), std::back_inserter(*result));

  return result;
} // end PrimitiveApi::surfacesList()

Scene *PrimitiveApi
  ::scene(Gotham::AttributeMap &attr)
{
  Scene *result = 0;

  size_t numThreads = lexical_cast<size_t>(attr["renderer:threads"]);

  if(numThreads > 1)
  {
    result = new RasterizableScene<CudaScene>();
  } // end if
  else
  {
    if(attr["scene:castshadows"] == std::string("false"))
    {
      result = new RasterizableScene<UnshadowedScene>();
    } // end if
    else
    {
      result = new RasterizableScene<Scene>();
    } // end else
  } // end else

  return result;
} // end PrimitiveApi::scene()

