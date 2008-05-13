/*! \file DebugRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of DebugRenderer class.
 */

#include "DebugRenderer.h"
#include "../geometry/Point.h"
#include "../geometry/Vector.h"
#include "../geometry/Ray.h"
#include "../primitives/Scene.h"
#include "../records/RenderFilm.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "../primitives/SurfacePrimitive.h"
#include "../primitives/SurfacePrimitiveList.h"

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
  float2 step(1.0f / film->getWidth(),
              1.0f / film->getHeight());

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

  progress.restart(film->getWidth() * film->getHeight());
  for(size_t y = 0;
      y < film->getHeight();
      ++y, uv[1] += step[1])
  {
    uv[0] = 0;
    for(size_t x = 0;
        x < film->getWidth();
        ++x, uv[0] += step[0])
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

      film->deposit(x,y,L);

      // purge all malloc'd memory for this sample
      mShadingContext->freeAll();

      ++progress;
    } // end for x
  } // end for y
} // end DebugRenderer::kernel()

