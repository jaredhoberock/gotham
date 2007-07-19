/*! \file Scene.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Scene.h.
 */

#include "Scene.h"

Scene
  ::Scene(void)
    :mPrimitive(),mRaysCast(0)
{
  ;
} // end Scene::Scene()

Scene
  ::Scene(boost::shared_ptr<Primitive> p)
    :mPrimitive(),mRaysCast(0)
{
  setPrimitive(p);
} // end Scene::Scene()

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
  ::resetRaysCast(void)
{
  mRaysCast = 0;
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

