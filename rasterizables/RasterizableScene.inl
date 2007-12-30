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
  // first rasterize the scene at large
  Rasterizable *r = dynamic_cast<Rasterizable*>(Parent0::mPrimitive.get());
  if(r != 0)
  {
    r->rasterize();
  } // end if

  // now turn off lighting, and rasterize the emitters in yellow
  r = dynamic_cast<Rasterizable*>(Parent0::mEmitters.get());
  if(r != 0)
  {
    glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_LIGHTING);
    glDepthFunc(GL_LEQUAL);
    glColor3f(1,1,0);
    r->rasterize();
    glPopAttrib();
  } // end if

  // now turn off lighting, and rasterize the sensors in green
  r = dynamic_cast<Rasterizable*>(Parent0::mSensors.get());
  if(r != 0)
  {
    glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_LIGHTING);
    glDepthFunc(GL_LEQUAL);
    glColor3f(0,1,0);
    r->rasterize();
    glPopAttrib();
  } // end if
} // end RasterizableScene::rasterize()

