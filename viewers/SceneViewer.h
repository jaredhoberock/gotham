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
#include <commonviewer/CommonViewer.h>
#include "../primitives/Scene.h"
#include <boost/shared_ptr.hpp>

#if USE_QGLVIEWER
#include <QGLViewer/qglviewer.h>
#include <QKeyEvent>
#else
#include <glutviewer/GlutViewer.h>
#endif // USE_QGLVIEWER

class SceneViewer
#if USE_QGLVIEWER
  : public CommonViewer<QGLViewer,QKeyEvent,QString,qglviewer::Vec>
#else
  : public CommonViewer<GlutViewer,KeyEvent,std::string,gpcpu::float3>
#endif // USE_QGLVIEWER
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
#if USE_QGLVIEWER
    typedef CommonViewer<QGLViewer,QKeyEvent,QString,qglviewer::Vec> Parent;
#else
    typedef CommonViewer<GlutViewer,KeyEvent,std::string,gpcpu::float3> Parent;
#endif // USE_QGLVIEWER

    /*! This method sets mScene.
     *  \param s Sets mScene.
     */
    inline virtual void setScene(boost::shared_ptr<Scene> s);

    /*! This method calls mRasterizeScene().
     */
    inline virtual void draw(void);

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

