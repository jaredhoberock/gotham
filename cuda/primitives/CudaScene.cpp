/*! \file CudaScene.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaScene class.
 */

#include "CudaScene.h"
#include "CudaPrimitive.h"

using namespace stdcuda;

void CudaScene
  ::intersect(device_ptr<const float4> originsAndMinT,
              device_ptr<const float4> directionsAndMaxT,
              device_ptr<CudaIntersection> intersections,
              device_ptr<int> stencil,
              const size_t n) const
{
  mRaysCast += n;
  const CudaPrimitive *prim = dynamic_cast<const CudaPrimitive*>(getPrimitive().get());
  prim->intersect(originsAndMinT, directionsAndMaxT, intersections, stencil, n);
} // end CudaScene::intersect()

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

