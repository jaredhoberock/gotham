/*! \file SceneRasterizer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Rasterizer
 *         for rasterizing an entire Scene.
 */

#ifndef SCENE_RASTERIZER_H
#define SCENE_RASTERIZER_H

#include "../primitives/Scene.h"
#include "Rasterizer.h"
#include "MeshRasterizer.h"

// FIXME: just rasterizes meshes for now
class SceneRasterizer
  : public Rasterizer<Scene>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Rasterizer<Scene> Parent;

    /*! Null constructor does nothing.
     */
    SceneRasterizer(void);

    SceneRasterizer(boost::shared_ptr<Scene> s);
    virtual void operator()(void);
    virtual void setPrimitive(boost::shared_ptr<Scene> s);

  protected:
    std::vector<MeshRasterizer> mMeshRasterizers;
}; // end SceneRasterizer

#endif // SCENE_RASTERIZER_H

