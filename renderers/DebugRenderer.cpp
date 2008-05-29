/*! \file DebugRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of DebugRenderer class.
 */

#include "DebugRenderer.h"
#include "../include/Point.h"
#include "../include/Vector.h"
#include "../geometry/Ray.h"
#include "../primitives/Scene.h"
#include "../records/RenderFilm.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "../primitives/SurfacePrimitive.h"
#include "../primitives/SurfacePrimitiveList.h"
#include <hilbertsequence/HilbertSequence.h>

using namespace boost;
using namespace gpcpu;

DebugRenderer
  ::DebugRenderer(void)
{
  ;
} // end DebugRenderer::DebugRenderer()

DebugRenderer
  ::DebugRenderer(shared_ptr<const Scene> s,
                  shared_ptr<Record> r)
    :Parent(s,r)
{
  ;
} // end DebugRenderer::DebugRenderer()

void DebugRenderer
  ::kernel(ProgressCallback &progress)
{
  // XXX TODO: kill this
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());

  size_t xStrata = 4;
  size_t yStrata = 4;
  HilbertSequence seq(0.0f, 1.0f,
                      0.0f, 1.0f,
                      xStrata * film->getWidth(),
                      yStrata * film->getHeight());

  Ray r;
  Point o;
  ::Vector d;
  Normal n;
  float pdf;
  bool delta;
  Intersection inter;
  float2 uv(0,0); 

  // sample from the list of sensors
  const SurfacePrimitive *sensor = 0;
  float temp;
  mScene->getSensors()->sampleSurfaceArea(0.0f, &sensor, temp);

  float weight = 1.0f / (xStrata * yStrata);

  progress.restart(film->getWidth() * film->getHeight() * xStrata * yStrata);

  while(seq(uv[0], uv[1]))
  {
    // sample a point on the sensor
    DifferentialGeometry dgSensor;
    sensor->sampleSurfaceArea(0,0,0,dgSensor,pdf);

    // generate a Ray
    ScatteringDistributionFunction &s = *mShadingContext->evaluateSensor(sensor->getMaterial(), dgSensor);

    // sample a sensing direction
    s.sample(dgSensor, uv[0], uv[1], 0.5f, d, pdf, delta);
    r = Ray(dgSensor.getPoint(), d);

    Spectrum L(0.0f,0.0f,0.0f);

    // intersect the Scene
    if(mScene->intersect(r, inter))
    {
      PrimitiveHandle prim = inter.getPrimitive();
      const SurfacePrimitive *sp = static_cast<const SurfacePrimitive*>((*mScene->getPrimitives())[prim].get());
      ScatteringDistributionFunction *f = mShadingContext->evaluateScattering(sp->getMaterial(), inter.getDifferentialGeometry());
      L = f->evaluate(-d,inter.getDifferentialGeometry(),-d);

      // add emission
      ScatteringDistributionFunction *e = mShadingContext->evaluateEmission(sp->getMaterial(), inter.getDifferentialGeometry());
      L += e->evaluate(-d, inter.getDifferentialGeometry());
    } // end if

    film->deposit(uv[0],uv[1], weight * L);

    // purge all malloc'd memory for this sample
    mShadingContext->freeAll();

    ++progress;
  } // end while
} // end DebugRenderer::kernel()

