/*! \file Scene.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Scene class.
 */

#include "Scene.h"

Scene
  ::~Scene(void)
{
  ;
} // end Scene::~Scene()

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
  return getPrimitive()->intersect(r);
} // end Scene::intersect()

void Scene
  ::getBoundingBox(BoundingBox &b) const
{
  return mPrimitive->getBoundingBox(b);
} // end Scene::getBoundingBox()

