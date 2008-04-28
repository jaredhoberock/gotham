/*! \file CudaScene.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaScene class.
 */

#include "CudaScene.h"
#include "CudaPrimitive.h"

using namespace stdcuda;

void CudaScene
  ::intersect(const device_ptr<const float4> &originsAndMinT,
              const device_ptr<const float4> &directionsAndMaxT,
              const device_ptr<CudaIntersection> &intersections,
              const device_ptr<bool> &stencil,
              const size_t n) const
{
  mRaysCast += n;
  const CudaPrimitive *prim = dynamic_cast<const CudaPrimitive*>(getPrimitive().get());
  prim->intersect(originsAndMinT, directionsAndMaxT, intersections, stencil, n);
} // end CudaScene::intersect()

void CudaScene
  ::shadow(const device_ptr<const float4> &originsAndMinT,
           const device_ptr<const float4> &directionsAndMaxT,
           const device_ptr<const bool> &stencil,
           const device_ptr<bool> &results,
           const size_t n) const
{
  const CudaPrimitive *prim = dynamic_cast<const CudaPrimitive*>(getPrimitive().get());
  prim->intersect(originsAndMinT, directionsAndMaxT, stencil, results, n);

  // XXX reductions here
  // mRaysCast += reduce(stencil)
  // mShadowRaysCast += reduce(stencil)
  // mBlockedShadowRays += reduce(result)
} // end CudaScene::shadow()

void CudaScene
  ::setPrimitive(boost::shared_ptr<Primitive> g)
{
  // we require that the Primitive also be a CudaPrimitive
  if(dynamic_cast<CudaPrimitive*>(g.get()))
  {
    Parent::setPrimitive(g);
  } // end if
  else
  {
    std::cerr << "CudaScene::setPrimitive(): primitive must be a CudaPrimitive." << std::endl;
    exit(-1);
  } // end CudaScene::setPrimitive()
} // end CudaScene::setPrimitive()

