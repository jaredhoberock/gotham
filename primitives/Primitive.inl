/*! \file Primitive.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Primitive.h.
 */

#include "Primitive.h"
#include "../geometry/Ray.h"

Primitive
  ::Primitive(void)
    :mName("")
{
  ;
} // end Primitive::Primitive()

Primitive
  ::~Primitive(void)
{
  ;
} // end Primitive::~Primitive()

bool Primitive
  ::intersect(Ray &r, Intersection &inter) const
{
  return false;
} // end Primitive::intersect()

void Primitive
  ::intersect(Ray *rays,
              Intersection *intersections,
              bool *stencil,
              const size_t n) const
{
  Ray *r = rays;
  Intersection *inter = intersections;
  bool *s = stencil;
  Ray *end = rays + n;
  for(;
      r != end;
      ++r, ++inter, ++s)
  {
    *s = intersect(*r, *inter);
  } // end for i
} // end Primitive::intersect()

bool Primitive
  ::intersect(const Ray &r) const
{
  return false;
} // end Primitive::intersect()

const std::string &Primitive
  ::getName(void) const
{
  return mName;
} // end Primitive::getName()

void Primitive
  ::setName(const std::string &name)
{
  mName = name;
  mNameHash = mStringHasher(name);
} // end Primitive::setName()

size_t Primitive
  ::getNameHash(void) const
{
  return mNameHash;
} // end Primitive::getNameHash()

void Primitive
  ::finalize(void)
{
  ;
} // end Primitive::finalize()

void Primitive
  ::setPrimitiveHandle(const PrimitiveHandle p)
{
  mPrimitiveHandle = p;
} // end Primitive::setPrimitiveHandle()

PrimitiveHandle Primitive
  ::getPrimitiveHandle(void) const
{
  return mPrimitiveHandle;
} // end PrimitiveHandle::getPrimitiveHandle()

