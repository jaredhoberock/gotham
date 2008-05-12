/*! \file CudaPrimitiveApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaPrimitiveApi class.
 */

#include "CudaPrimitiveApi.h"
#include "CudaTriangleBVH.h"
#include "CudaScene.h"
#include "../../rasterizables/RasterizableScene.h"
#include "../../rasterizables/RasterizablePrimitiveList.h"
#include <boost/lexical_cast.hpp>

using namespace boost;

PrimitiveList *CudaPrimitiveApi
  ::list(Gotham::AttributeMap &attr,
         const PrimitiveList &prims)
{
  PrimitiveList *result = 0;

  // CudaTriangleBVH needs to know the number of threads
  size_t numThreads = lexical_cast<size_t>(attr["renderer:threads"]);

  CudaTriangleBVH *bvh = new RasterizablePrimitiveList<CudaTriangleBVH>();
  bvh->setWorkBatchSize(numThreads);
  result = bvh;

  // copy the prims
  std::copy(prims.begin(), prims.end(), std::back_inserter(*result));

  return result;
} // end PrimitiveApi::list()

SurfacePrimitiveList *CudaPrimitiveApi
  ::surfacesList(Gotham::AttributeMap &attr,
                 const SurfacePrimitiveList &surfaces)
{
  SurfacePrimitiveList *result = 0;

  // CudaTriangleBVH needs to know the number of threads
  size_t numThreads = lexical_cast<size_t>(attr["renderer:threads"]);

  // XXX no need for this to be a bvh
  CudaTriangleBVH *bvh = new RasterizablePrimitiveList<CudaTriangleBVH>();
  bvh->setWorkBatchSize(numThreads);
  result = bvh;

  // copy the prims
  std::copy(surfaces.begin(), surfaces.end(), std::back_inserter(*result));

  return result;
} // end PrimitiveApi::surfacesList()

CudaScene *CudaPrimitiveApi
  ::scene(Gotham::AttributeMap &attr)
{
  CudaScene *result = 0;

  result = new RasterizableScene<CudaScene>();
} // end CudaPrimitiveApi::scene()

