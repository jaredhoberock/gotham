/*! \file Scene.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Scene.h.
 */

#include "Scene.h"
#include "PrimitiveList.h"
#include "SurfacePrimitiveList.h"

Scene
  ::Scene(void)
    :mPrimitive()
{
  resetStatistics();
} // end Scene::Scene()

Scene
  ::Scene(boost::shared_ptr<Primitive> p)
    :mPrimitive()
{
  resetStatistics();
  setPrimitive(p);
} // end Scene::Scene()

Scene
  ::~Scene(void)
{
  ;
} // end Scene::~Scene()

boost::shared_ptr<Primitive> Scene
  ::getPrimitive(void) const
{
  return mPrimitive;
} // end Scene::getGeometry()

void Scene
  ::setPrimitive(boost::shared_ptr<Primitive> p)
{
  mPrimitive = p;
} // end Scene::setGeometry()

long unsigned int Scene
  ::getRaysCast(void) const
{
  return mRaysCast;
} // end Scene::getRaysCast()

void Scene
  ::setRaysCast(const long unsigned int r) const
{
  mRaysCast = r;
} // end Scene::setRaysCast()

long unsigned int Scene
  ::getShadowRaysCast(void) const
{
  return mShadowRaysCast;
} // end Scene::getShadowRaysCast()

long unsigned int Scene
  ::getBlockedShadowRays(void) const
{
  return mBlockedShadowRays;
} // end Scene::getBlockedShadowRays()

void Scene
  ::resetStatistics(void)
{
  mRaysCast = 0;
  mShadowRaysCast = 0;
  mBlockedShadowRays = 0;
} // end Scene::resetRaysCast()

void Scene
  ::setEmitters(boost::shared_ptr<SurfacePrimitiveList> e)
{
  mEmitters = e;
} // end Scene::setEmitters()

void Scene
  ::setSensors(boost::shared_ptr<SurfacePrimitiveList> s)
{
  mSensors = s;
} // end Scene::setSensors()

void Scene
  ::setSurfaces(boost::shared_ptr<SurfacePrimitiveList> s)
{
  mSurfaces = s;
} // end Scene::setSensors()

void Scene
  ::setPrimitives(boost::shared_ptr<PrimitiveList> p)
{
  mPrimitives = p;
} // end Scene::setPrimitives()

const SurfacePrimitiveList *Scene
  ::getEmitters(void) const
{
  return mEmitters.get();
} // end Scene::getEmitters()

const SurfacePrimitiveList *Scene
  ::getSensors(void) const
{
  return mSensors.get();
} // end Scene::getSensors()

const SurfacePrimitiveList *Scene
  ::getSurfaces(void) const
{
  return mSurfaces.get();
} // end Scene::getSurfaces()

const PrimitiveList *Scene
  ::getPrimitives(void) const
{
  return mPrimitives.get();
} // end Scene::getPrimitives()

bool Scene
  ::intersect(Ray &r, Intersection &inter) const
{
  ++mRaysCast;
  return getPrimitive()->intersect(r,inter);
} // end Scene::intersect()

void Scene
  ::intersect(Ray *rays,
              Intersection *intersections,
              bool *stencil,
              const size_t n) const
{
  mRaysCast += n;
  return getPrimitive()->intersect(rays, intersections, stencil, n);
} // end Scene::intersect()

bool Scene
  ::intersect(const Ray &r) const
{
  ++mRaysCast;
  ++mShadowRaysCast;
  bool result = getPrimitive()->intersect(r);
  mBlockedShadowRays += result;
  return result;
} // end Scene::intersect()

void Scene
  ::getBoundingBox(BoundingBox &b) const
{
  return mPrimitive->getBoundingBox(b);
} // end Scene::getBoundingBox()

void Scene
  ::preprocess(void)
{
  // XXX we should rename this method to preprocess
  mEmitters->finalize();
  mSensors->finalize();

  // XXX this is broken, because SurfacePrimitiveList::finalize()
  //     will set the PrimitiveHandles of each primitive, when
  //     actually they correspond to an index in mPrimitives
  mSurfaces->finalize();
  mPrimitive->finalize();
} // end Scene::preprocess()

