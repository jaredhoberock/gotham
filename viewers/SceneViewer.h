/*! \file SceneViewer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a viewer for
 *         Gotham Scenes.
 */

#ifndef SCENE_VIEWER_H
#define SCENE_VIEWER_H

#include <GL/glew.h>
#include <QGLViewer/qglviewer.h>
#include <QKeyEvent>
#include <commonviewer/CommonViewer.h>
#include "../primitives/Scene.h"
#include "../rasterizers/SceneRasterizer.h"
#include <boost/shared_ptr.hpp>

class Camera;

class SceneViewer
  : public CommonViewer<QGLViewer,QKeyEvent>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef CommonViewer<QGLViewer,QKeyEvent> Parent;

    /*! This method sets mScene.
     *  \param s Sets mScene.
     */
    inline virtual void setScene(boost::shared_ptr<Scene> s);

    /*! This method calls mRasterizeScene().
     */
    inline virtual void draw(void);

    inline virtual void init(void);

  protected:
    inline virtual void getCamera(Camera &cam) const;

    boost::shared_ptr<Scene> mScene;
    SceneRasterizer mRasterizeScene;
}; // end SceneViewer

#include "SceneViewer.inl"

#endif // SCENE_VIEWER_H

