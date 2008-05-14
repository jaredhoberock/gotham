/*! \file SceneViewer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for SceneViewer.h.
 */

#include "SceneViewer.h"
#include "../rasterizables/Rasterizable.h"
#include "../geometry/BoundingBox.h"
#include "../primitives/SurfacePrimitiveList.h"
#include "../shading/PerspectiveSensor.h"
#include <fstream>
#include <qfiledialog.h>

void SceneViewer
  ::setScene(boost::shared_ptr<Scene> s)
{
  mScene = s;

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
  Rasterizable *r = dynamic_cast<Rasterizable*>(mScene.get());
  if(r != 0)
  {
    r->rasterize();
  } // end if
} // end SceneViewer::draw()

void SceneViewer
  ::keyPressEvent(Parent::KeyEvent *e)
{
  switch(e->key())
  {
    case 'C':
    {
      printCameraCode();
      break;
    } // end case 'c'

    default:
    {
      Parent::keyPressEvent(e);
      break;
    } // end default
  } // end switch
} // end SceneViewer::keyPressEvent()

void SceneViewer
  ::printCameraCode(void) const
{
  Point eye(camera()->position());
  Vector look(camera()->viewDirection());
  Vector lookAt = eye + look;
  Vector up(camera()->upVector());

  std::cerr << "PushMatrix()" << std::endl;
  std::cerr << "LookAt([" << eye[0] << "," << eye[1] << "," << eye[2] << "]," << std::endl;
  std::cerr << "       [" << lookAt[0] << "," << lookAt[1] << "," << lookAt[2] << "]," << std::endl;
  std::cerr << "       [" << up[0] << "," << up[1] << "," << up[2] << "])" << std::endl;
  std::cerr << "Camera(" << camera()->aspectRatio() << "," << camera()->fieldOfView() << "," << camera()->zNear() << ")" << std::endl;
  std::cerr << "PopMatrix()" << std::endl;
} // end SceneViewer::printCameraCode()

