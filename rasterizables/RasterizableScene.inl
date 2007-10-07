/*! \file RasterizableScene.inl
 *  \author Jared Hoberock
 *  \brief Inline file for RasterizableScene.h.
 */

#include "RasterizableScene.h"

template<typename SceneParentType>
  RasterizableScene<SceneParentType>
    ::RasterizableScene(void)
      :Parent0(),Parent1()
{
  ;
} // end RasterizableScene::RasterizableScene()

template<typename SceneParentType>
  RasterizableScene<SceneParentType>
    ::RasterizableScene(boost::shared_ptr<Primitive> p)
      :Parent0(p),Parent1()
{
  ;
} // end RasterizableScene::RasterizableScene()

template<typename SceneParentType>
  void RasterizableScene<SceneParentType>
    ::rasterize(void)
{
  Rasterizable *r = dynamic_cast<Rasterizable*>(Parent0::mPrimitive.get());
  if(r != 0)
  {
    r->rasterize();
  } // end if
} // end RasterizableScene::rasterize()

