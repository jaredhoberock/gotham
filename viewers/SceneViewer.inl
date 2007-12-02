/*! \file SceneViewer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for SceneViewer.h.
 */

#include "SceneViewer.h"
#include "../rasterizables/Rasterizable.h"
#include <pricksie/geometry/BoundingBox.h>
#include "../primitives/SurfacePrimitiveList.h"
#include "../geometry/DifferentialGeometry.h"
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
    case 'c':
    {
      qglviewer::Vec v = camera()->position();
      std::cerr << "eye: " << v[0] << "," << v[1] << "," << v[2] << std::endl;
      v += camera()->viewDirection();
      std::cerr << "look at: " << v[0] << "," << v[1] << "," << v[2] << std::endl;
      v = camera()->upVector();
      std::cerr << "up: " << v[0] << "," << v[1] << "," << v[2] << std::endl;
      break;
    } // end case 'c'

    default:
    {
      Parent::keyPressEvent(e);
      break;
    } // end default
  } // end switch
} // end SceneViewer::keyPressEvent()

