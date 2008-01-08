/*! \file Primitive.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Primitive.h.
 */

#include "Primitive.h"

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

const DifferentialGeometry &Primitive::Intersection
  ::getDifferentialGeometry(void) const
{
  return mDifferentialGeometry;
} // end Primitive::Intersection::getDifferentialGeometry()

void Primitive::Intersection
  ::setDifferentialGeometry(const DifferentialGeometry &dg)
{
  mDifferentialGeometry = dg;
} // end Primitive::Intersection::setDifferentialGeometry()

const Primitive *Primitive::Intersection
  ::getPrimitive(void) const
{
  return mPrimitive;
} // end Primitive::Intersection::getPrimitive()

void Primitive::Intersection
  ::setPrimitive(const Primitive *p)
{
  mPrimitive = p;
} // end Primitive::Intersetion::setPrimitive()


bool Primitive
  ::intersect(Ray &r, Intersection &inter) const
{
  return false;
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

