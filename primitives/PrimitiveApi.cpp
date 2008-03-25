/*! \file PrimitiveApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PrimitiveApi class.
 */

#include "PrimitiveApi.h"
#include "../rasterizables/RasterizablePrimitiveList.h"
#include "../primitives/PrimitiveBSP.h"
#include "TriangleBVH.h"
#include "CUDATriangleBVH.h"
#include <algorithm>

using namespace boost;

PrimitiveList<> *PrimitiveApi
  ::list(Gotham::AttributeMap &attr,
         const PrimitiveList<> &prims)
{
  PrimitiveList<> *result = 0;

  size_t numThreads = lexical_cast<size_t>(attr["renderer:threads"]);

  if(numThreads > 1)
  {
    TriangleBVH *bvh = new RasterizablePrimitiveList< TriangleBVH >();
    //CUDATriangleBVH *bvh = new RasterizablePrimitiveList<CUDATriangleBVH>();
    //bvh->setWorkBatchSize(numThreads);
    result = bvh;
  } // end if
  else
  {
    result = new RasterizablePrimitiveList< PrimitiveBSP<> >();
  } // end else

  // copy the prims
  std::copy(prims.begin(), prims.end(), std::back_inserter(*result));

  return result;
} // end PrimitiveApi::list()

