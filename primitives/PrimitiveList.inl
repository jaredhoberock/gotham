/*! \file PrimitiveList.inl
 *  \author Jared Hoberock
 *  \brief Inline file for PrimitiveList.h.
 */

#include "PrimitiveList.h"
#include "../geometry/BoundingBox.h"

template<typename PrimitiveType>
  void PrimitiveList<PrimitiveType>
    ::clear(void)
{
  Parent1::clear();
  mBoundingBox.setEmpty();
} // end PrimitiveList::clear()

template<typename PrimitiveType>
  void PrimitiveList<PrimitiveType>
    ::push_back(boost::shared_ptr<PrimitiveType> &p)
{
  Parent1::push_back(p);

  BoundingBox b;
  p->getBoundingBox(b);
  mBoundingBox.addPoint(b[0]);
  mBoundingBox.addPoint(b[1]);
} // end PrimitiveList::push_back()

template<typename PrimitiveType>
  void PrimitiveList<PrimitiveType>
    ::getBoundingBox(BoundingBox &b) const
{
  b = mBoundingBox;
} // end PrimitiveList::getBoundingBox()

template<typename PrimitiveType>
  bool PrimitiveList<PrimitiveType>
    ::intersect(Ray &r, Intersection &inter) const
{
  bool result = false;
  for(typename Parent1::const_iterator p = Parent1::begin();
      p != Parent1::end();
      ++p)
  {
    result |= (*p)->intersect(r,inter);
  } // end for p

  return result;
} // end PrimitiveList::intersect()

template<typename PrimitiveType>
  bool PrimitiveList<PrimitiveType>
    ::intersect(const Ray &r) const
{
  for(typename Parent1::const_iterator p = Parent1::begin();
      p != Parent1::end();
      ++p)
  {
    if((*p)->intersect(r)) return true;
  } // end for p

  return false;
} // end PrimitiveList::intersect()

