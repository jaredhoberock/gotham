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

  // hand to rasterizer
  mRasterizeScene.setPrimitive(s);

  // fit the view to the scene
  BoundingBox b;
  mScene->getBoundingBox(b);
  setSceneBoundingBox(qglviewer::Vec(b[0]),
                      qglviewer::Vec(b[1]));

  //// try to set the camera
  //const SurfacePrimitiveList *sensors = mScene->getSensors();
  //if(sensors != 0 && sensors->size() > 0)
  //{
  //  const SurfacePrimitive *surf = 0;
  //  float pdf;
  //  DifferentialGeometry dg;
  //  sensors->sampleSurfaceArea(0.5f, 0.5f, 0.5f, 0.5f,
  //                             &surf, dg, pdf);

  //  ScatteringDistributionFunction *f = surf->getMaterial()->evaluateSensor(dg);

  //  PerspectiveSensor *ps = dynamic_cast<PerspectiveSensor*>(f);
  //  if(ps != 0)
  //  {
  //    Vector right, up;
  //    right = ps->getRight();
  //    up = ps->getUp();
  //    Vector look = -right.cross(up);

  //    // use the differential geometry to try to position the camera
  //    camera()->setPosition(qglviewer::Vec(dg.getPoint()));
  //    camera()->setUpVector(qglviewer::Vec(up));
  //    camera()->setViewDirection(qglviewer::Vec(look));
  //  } // end if

  //  ScatteringDistributionFunction::mPool.freeAll();
  //} // end if
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
  Rasterizable *r = dynamic_cast<Rasterizable*>(mScene.get());
  if(r != 0)
  {
    r->rasterize();
  } // end if
} // end SceneViewer::draw()


