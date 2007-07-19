/*! \file SceneViewer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for SceneViewer.h.
 */

#include "SceneViewer.h"
#include "../Camera.h"
#include <pricksie/geometry/BoundingBox.h>
#include <fstream>
#include <qfiledialog.h>

void SceneViewer
  ::setScene(boost::shared_ptr<Scene> s)
{
  mScene = s;

  // hand to rasterizer
  mRasterizeScene.setPrimitive(s);

  // fit the view to the scene
  BoundingBox b;
  mScene->getBoundingBox(b);
  setSceneBoundingBox(qglviewer::Vec(b[0]),
                      qglviewer::Vec(b[1]));
} // end SceneViewer::setScene()

void SceneViewer
  ::init(void)
{
  glewInit();
  Parent::init();
} // end SceneViewer::init()

void SceneViewer
  ::draw(void)
{
  mRasterizeScene();
} // end SceneViewer::draw()

void SceneViewer
  ::getCamera(Camera &camera) const
{
  Vector3 up(QGLViewer::camera()->upVector()[0],
             QGLViewer::camera()->upVector()[1],
             QGLViewer::camera()->upVector()[2]);
  Vector3 look(QGLViewer::camera()->viewDirection()[0],
               QGLViewer::camera()->viewDirection()[1],
               QGLViewer::camera()->viewDirection()[2]);
  Point eye(QGLViewer::camera()->position()[0],
             QGLViewer::camera()->position()[1],
             QGLViewer::camera()->position()[2]);
  float fovy = QGLViewer::camera()->fieldOfView();

  // create a Camera
  camera = Camera(static_cast<float>(width()) / static_cast<float>(height()),
                  fovy,
                  //0.02f,
                  0,
                  eye,
                  look,
                  up);
} // end SceneViewer::getCamera()

