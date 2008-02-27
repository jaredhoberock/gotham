/*! \file PrimitiveBSP.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PrimitiveBSP class.
 */

#include "PrimitiveBSP.h"
#include "../geometry/Ray.h"

template<typename PrimitiveType>
  PrimitiveBSP<PrimitiveType>
    ::PrimitiveBSP(void)
      :Parent0(),Parent1()
{
  ;
} // end PrimitiveBSP::PrimitiveBSP()

template<typename PrimitiveType>
  float PrimitiveBSP<PrimitiveType>::Bounder
    ::operator()(unsigned int axis,
                 bool min,
                 const Primitive *p) const
{
  BoundingBox b;
  p->getBoundingBox(b);
  const Point &corner = (min)?(b.getMinCorner()):(b.getMaxCorner());
  return corner[axis];
} // end PrimitiveBounder::operator()()

template<typename PrimitiveType>
  void PrimitiveBSP<PrimitiveType>
    ::finalize(void)
{
  // build the bsp
  Bounder bounder;
  std::vector<const PrimitiveType*> temp;
  for(typename Parent0::const_iterator i = Parent0::begin();
      i != Parent0::end();
      ++i)
  {
    temp.push_back((*i).get());
  } // end for i

  Parent1::buildTree(temp.begin(),temp.end(),bounder);
} // end PrimitiveBSP::finalize()

template<typename PrimitiveType>
  bool PrimitiveBSP<PrimitiveType>::Shadower
    ::operator()(const Point &anchor, const Point &dir,
                 const Primitive **begin,
                 const Primitive **end,
                 float minT, float maxT)
{
  // XXX PERF it's really wasteful to construct this extra Ray
  // XXX DESIGN consider another standard intersection interface
  //     that doesn't take a Ray but just the Ray members
  Ray r(anchor, dir, minT, maxT);
  while(begin != end)
  {
    // intersect ray with object
    if((*begin)->intersect(r))
    {
      return true;
    } // end if

    ++begin; 
  } // end while

  return false;
} // end Shadower::operator()()

template<typename PrimitiveType>
  bool PrimitiveBSP<PrimitiveType>
    ::intersect(const Ray &r) const
{
  // create a Shadower
  Shadower shadower;
  if(Parent1::shadow(r.getAnchor(), r.getDirection(), r.getInterval()[0], r.getInterval()[1], shadower))
  {
    return true;
  } // end if

  return false;
} // end PrimitiveBSP::intersect()

template<typename PrimitiveType>
  bool PrimitiveBSP<PrimitiveType>::Intersector
    ::operator()(const Point &anchor,
                 const Point &dir,
                 const Primitive** begin,
                 const Primitive** end,
                 float minT, float maxT)
{
  bool result = false;
  // XXX PERF same critique here as above
  Ray r(anchor, dir, minT, maxT);
  Primitive::Intersection intersection;

  while(begin != end)
  {
    // intersect ray with object
    if((*begin)->intersect(r, intersection))
    {
      mIntersection = intersection;
      mHitTime = r.getInterval()[1];
      result = true;

      // XXX this is getting annoying
      //     log this failure instead of an assert
      //assert(r.getInterval()[0] <= r.getInterval()[1]);
    } // end if

    ++begin;
  } // end while

  return result;
} // end Intersector::operator()()

template<typename PrimitiveType>
  bool PrimitiveBSP<PrimitiveType>
    ::intersect(Ray &r, typename Parent0::Intersection &inter) const
{
  // create an intersector
  Intersector intersector;
  if(Parent1::intersect(r.getAnchor(), r.getDirection(), r.getInterval()[0], r.getInterval()[1], intersector))
  {
    inter = intersector.mIntersection;
    r.getInterval()[1] = intersector.mHitTime;
    return true;
  } // end if

  return false;
} // end PrimitiveBSP::intersect()

