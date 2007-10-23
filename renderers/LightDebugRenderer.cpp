/*! \file LightDebugRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of LightDebugRenderer class.
 */

#include "LightDebugRenderer.h"
#include "../geometry/Point.h"
#include "../geometry/Vector.h"
#include "../geometry/Ray.h"
#include "../primitives/Scene.h"
#include "../primitives/SurfacePrimitiveList.h"
#include "../records/RenderFilm.h"
#include "../shading/Material.h"
#include "../shading/FunctionAllocator.h"
#include "../shading/ScatteringDistributionFunction.h"
#include "../primitives/SurfacePrimitive.h"
#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/uniform_01.hpp>

using namespace boost;
using namespace gpcpu;

LightDebugRenderer
  ::LightDebugRenderer(void)
{
  ;
} // end LightDebugRenderer::LightDebugRenderer()

LightDebugRenderer
  ::LightDebugRenderer(shared_ptr<const Scene> s,
                       shared_ptr<RenderFilm> f)
    :Parent(s,f)
{
  ;
} // end LightDebugRenderer::LightDebugRenderer()

void LightDebugRenderer
  ::kernel(ProgressCallback &progress)
{
  float2 step(1.0f / mFilm->getWidth(),
              1.0f / mFilm->getHeight());

  typedef boost::random::lagged_fibonacci_01<float, 48, 607, 273> Rng;
  Rng generator(13u);
  boost::uniform_01<Rng, float> uni01(generator);

  // sample from the list of sensors
  const SurfacePrimitive *sensor = 0;
  float temp;
  mScene->getSensors()->sampleSurfaceArea(0.0f, &sensor, temp);

  Ray r;
  Point o;
  Vector3 d;
  Normal n;
  float sensorSurfaceAreaPdf;
  bool sensorDelta;
  float sensorSolidAnglePdf;
  float lightPdf;
  Primitive::Intersection inter;
  float2 uv(0,0); 

  progress.restart(mFilm->getWidth() * mFilm->getHeight());
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
      sensor->sampleSurfaceArea(0,0,0,dgSensor,sensorSurfaceAreaPdf);

      // generate a Ray
      ScatteringDistributionFunction &s = *sensor->getMaterial()->evaluateSensor(dgSensor);

      // sample a sensing direction
      Spectrum sensorResponse = s.sample(dgSensor, uv[0], uv[1], 0.5f, d, sensorSolidAnglePdf, sensorDelta);
      r = Ray(dgSensor.getPoint(), d);

      Spectrum L(0.1f,0.1f,0.1f);

      // intersect the Scene
      if(mScene->intersect(r, inter))
      {
        const Primitive *prim = inter.getPrimitive();
        const SurfacePrimitive *surface = dynamic_cast<const SurfacePrimitive*>(prim);
        const Material *material = surface->getMaterial();
        const DifferentialGeometry &dg = inter.getDifferentialGeometry();

        // evaluate emission
        // XXX correctly account for the sensor's surface area pdf
        ScatteringDistributionFunction &e = *material->evaluateEmission(dg);
        L = sensorResponse * e(-d,dg) / (sensorSolidAnglePdf);

        ScatteringDistributionFunction &f = *material->evaluateScattering(dg);

        // sample light source
        float u0 = uni01(), u1 = uni01(), u2 = uni01(), u3 = uni01();
        const SurfacePrimitive *light;
        DifferentialGeometry dgLight;
        mScene->getEmitters()->sampleSurfaceArea(u0,u1,u2,u3,&light,dgLight, lightPdf);

        Point p = dg.getPoint();
        r.set(p, dgLight.getPoint());
        if(!mScene->intersect(r))
        {
          Vector3 l = r.getDirection();
          l = l.normalize();

          ScatteringDistributionFunction &e = *light->getMaterial()->evaluateEmission(dgLight);

          // XXX correctly account for the sensor's surface area pdf
          L += sensorResponse * f(-d,dg,l) * e(-l,dgLight) / (sensorSolidAnglePdf * lightPdf);
        } // end if
      } // end if

      mFilm->raster(x,y) = L;

      // purge all malloc'd memory for this sample
      ScatteringDistributionFunction::mPool.freeAll();

      ++progress;
    } // end for x
  } // end for y
} // end LightDebugRenderer::kernel()

