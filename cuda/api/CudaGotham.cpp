/*! \file CudaGotham.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaGotham class.
 */

#include "CudaGotham.h"
#include "../primitives/CudaScene.h"
#include "../primitives/CudaPrimitiveApi.h"
#include "../renderers/CudaRendererApi.h"
#include "../shading/CudaShadingApi.h"
#include <gotham/records/RecordApi.h>
#include <gotham/viewers/RenderViewer.h>

#include <boost/shared_ptr.hpp>
using namespace boost;

#pragma warning(push)
#pragma warning(disable : 4311 4312)
#include <Qt/qapplication.h>
#pragma warning(pop)

void CudaGotham
  ::render(void)
{
  AttributeMap &attr = mAttributeStack.back();

  // create a new Scene
  shared_ptr<Scene> s(CudaPrimitiveApi::scene(mAttributeStack.back()));

  // create a final PrimitiveList
  PrimitiveList *list = CudaPrimitiveApi::list(mAttributeStack.back(),
                                               *mPrimitives);

  // set every primitive's PrimitiveHandle
  // XXX this really sucks but i can't find a better solution
  PrimitiveHandle h = 0;
  for(PrimitiveList::iterator prim = list->begin();
      prim != list->end();
      ++prim, ++h)
  {
    (*prim)->setPrimitiveHandle(h);
  } // end for i

  // hand over the primitives
  shared_ptr<PrimitiveList> listPtr(list);
  s->setPrimitive(listPtr);
  s->setPrimitives(listPtr);

  // give the surfaces to the scene
  s->setSurfaces(mSurfaces);

  // create a final SurfacePrimitiveList for emitters
  shared_ptr<SurfacePrimitiveList> emitters(CudaPrimitiveApi::surfacesList(mAttributeStack.back(),
                                                                           *mEmitters));
  // give the emitters to the scene
  s->setEmitters(emitters);

  // create a final SurfacePrimitiveList for sensors
  shared_ptr<SurfacePrimitiveList> sensors(CudaPrimitiveApi::surfacesList(mAttributeStack.back(),
                                                                          *mSensors));
  // give the sensors to the scene
  s->setSensors(sensors);

  // create a new Renderer
  mRenderer.reset(CudaRendererApi::renderer(mAttributeStack.back(), mPhotonMaps));

  // create a Record
  shared_ptr<Record> record;
  record.reset(RecordApi::record(mAttributeStack.back()));

  // create a ShadingContext
  shared_ptr<ShadingContext> context;
  context.reset(CudaShadingApi::context(mAttributeStack.back(), mMaterials));

  // give everything to the renderer
  mRenderer->setScene(s);
  mRenderer->setRecord(record);
  mRenderer->setShadingContext(context);

  // headless render?
  bool headless = (attr["viewer"] == std::string("false"));

  if(!headless)
  {
    int zero = 0;
    QApplication application(zero,0);

    RenderViewer v;

    // title the window the name of the outfile
    v.setWindowTitle(attr["record:outfile"].c_str());

    // everything to the viewer
    v.setScene(s);

    v.setRenderer(mRenderer);

    v.setSnapshotFileName(mRenderer->getRenderParameters().c_str());

    v.setGamma(lexical_cast<float>(attr["viewer:gamma"]));

    // try to tell the viewer where to look
    // bail out otherwise
    try
    {
      float fovy   = lexical_cast<float>(attr["viewer:fovy"]);
      float eyex   = lexical_cast<float>(attr["viewer:eyex"]);
      float eyey   = lexical_cast<float>(attr["viewer:eyey"]);
      float eyez   = lexical_cast<float>(attr["viewer:eyez"]);
      float upx    = lexical_cast<float>(attr["viewer:upx"]);
      float upy    = lexical_cast<float>(attr["viewer:upy"]);
      float upz    = lexical_cast<float>(attr["viewer:upz"]);
      float lookx  = lexical_cast<float>(attr["viewer:lookx"]);
      float looky  = lexical_cast<float>(attr["viewer:looky"]);
      float lookz  = lexical_cast<float>(attr["viewer:lookz"]);
      float width  = lexical_cast<float>(attr["record:width"]);
      float height = lexical_cast<float>(attr["record:height"]);

      // convert degrees to radians
      float fovyRadians = fovy * PI / 180.0f;
      v.camera()->setFieldOfView(fovyRadians);
      v.camera()->setAspectRatio(width/height);
      v.camera()->setPosition(qglviewer::Vec(eyex,eyey,eyez));
      v.camera()->setUpVector(qglviewer::Vec(upx,upy,upz));
      v.camera()->setViewDirection(qglviewer::Vec(lookx,looky,lookz));
    } // end try
    catch(...)
    {
      ;
    } // end catch

    v.show();

    application.exec();
  } // end if
  else
  {
    // start rendering
    Renderer::ProgressCallback callback;
    mRenderer->render(callback);
  } // end else

  return;
} // end CudaGotham::render()

void CudaGotham
  ::sphere(const float cx,
           const float cy,
           const float cz,
           const float radius)
{
  std::cerr << "Warning: sphere() is not implemented." << std::endl;
} // end CudaGotham::sphere()

