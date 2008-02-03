/*! \file SceneViewer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a viewer for
 *         Gotham Scenes.
 */

#ifndef SCENE_VIEWER_H
#define SCENE_VIEWER_H

// glew.h will #include <windows.h>
#ifdef WIN32
#define NOMINMAX
#define WINDOWS_LEAN_AND_MEAN
#endif // WIN32

#include <GL/glew.h>
#include <QGLViewer/qglviewer.h>
#include <QKeyEvent>
#include <commonviewer/CommonViewer.h>
#include "../primitives/Scene.h"
#include <boost/shared_ptr.hpp>

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

    /*! This method outputs python code to position a camera
     *  at this SceneViewer's location.
     */
    inline void printCameraCode(void) const;

  protected:
    inline virtual void keyPressEvent(Parent::KeyEvent *e);

    boost::shared_ptr<Scene> mScene;
}; // end SceneViewer

#include "SceneViewer.inl"

#endif // SCENE_VIEWER_H

