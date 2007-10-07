/*! \file RasterizableScene.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Scene
 *         class which is also rasterizable.
 */

#ifndef RASTERIZABLE_SCENE_H
#define RASTERIZABLE_SCENE_H

#include "Rasterizable.h"
#include "../primitives/Scene.h"

template<typename SceneParentType>
  class RasterizableScene
    : public SceneParentType,
      public Rasterizable
{
  public:
    /*! \typedef Parent0
     *  \brief Shorthand.
     */
    typedef SceneParentType Parent0;

    /*! \typedef Parent1
     *  \brief Shorthand.
     */
    typedef Rasterizable Parent1;

    /*! Null constructor calls the Parents
     */
    inline RasterizableScene(void);

    /*! Constructor accepts a Primitive.
     *  \param p Sets Parent0::mPrimitive.
     */
    inline RasterizableScene(boost::shared_ptr<Primitive> p);

    /*! This method rasterizes Parent0::mPrimitive if it is
     *  an instance of Rasterizable.
     */
    inline virtual void rasterize(void);
}; // end RasterizableScene

#include "RasterizableScene.inl"

#endif // RASTERIZABLE_SCENE_H

