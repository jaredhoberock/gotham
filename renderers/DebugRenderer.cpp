/*! \file DebugRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of DebugRenderer class.
 */

#include "DebugRenderer.h"
#include "../geometry/Point.h"
#include "../geometry/Vector.h"
#include "../geometry/Ray.h"
#include "../primitives/Scene.h"
#include "../films/RandomAccessFilm.h"
#include "../shading/Material.h"
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
                  shared_ptr<RandomAccessFilm> f)
    :Parent(s,f)
{
  ;
} // end DebugRenderer::DebugRenderer()

void DebugRenderer
  ::render(ProgressCallback &progress)
{
  float2 step(1.0f / mFilm->getWidth(),
              1.0f / mFilm->getHeight());

  Ray r;
  Point o;
  Vector3 d;
  Normal n;
  float pdf;
  Primitive::Intersection inter;
  float2 uv(0,0); 
  unsigned int i = 0;
  unsigned int tenPercentSamples = mFilm->getWidth() * mFilm->getHeight() / 10;

  // sample from the list of sensors
  const SurfacePrimitive *sensor = 0;
  float temp;
  mScene->getSensors()->sampleSurfaceArea(0.0f, &sensor, temp);

  for(unsigned int y = 0;
      y < mFilm->getHeight();
      ++y, uv[1] += step[1])
  {
    uv[0] = 0;
    for(unsigned int x = 0;
        x < mFilm->getWidth();
        ++x, uv[0] += step[0])
    {
      // sample a point on the sensor
      DifferentialGeometry dgSensor;
      sensor->sampleSurfaceArea(0,0,0,dgSensor,pdf);

      // generate a Ray
      ScatteringDistributionFunction &s = *sensor->getMaterial()->evaluateSensor(dgSensor);

      // sample a sensing direction
      s.sample(dgSensor, uv[0], uv[1], 0.5f, d, pdf);
      r = Ray(dgSensor.getPoint(), d);

      Spectrum L(0.1f,0.1f,0.1f);

      // intersect the Scene
      if(mScene->intersect(r, inter))
      {
        const Primitive *prim = inter.getPrimitive();
        const SurfacePrimitive *surface = dynamic_cast<const SurfacePrimitive*>(prim);
        const Material *material = surface->getMaterial();
        ScatteringDistributionFunction *f = static_cast<const SurfacePrimitive*>(prim)->getMaterial()->evaluateScattering(inter.getDifferentialGeometry());
        L = f->evaluate(-d,inter.getDifferentialGeometry(),-d);
      } // end if

      mFilm->raster(x,y) = L;
      ++i;

      if(i % tenPercentSamples == 0)
      {
        progress(i);
      } // end if

      // purge all malloc'd memory for this sample
      ScatteringDistributionFunction::mPool.freeAll();
    } // end for x
  } // end for y
} // end DebugRenderer::render()

