/*! \file Scene.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Scene.h.
 */

#include "Scene.h"

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

bool Scene
  ::intersect(Ray &r, Primitive::Intersection &inter) const
{
  ++mRaysCast;
  return getPrimitive()->intersect(r,inter);
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

