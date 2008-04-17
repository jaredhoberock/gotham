/*! \file PrimitiveList.inl
 *  \author Jared Hoberock
 *  \brief Inline file for PrimitiveList.h.
 */

#include "PrimitiveList.h"
#include "../geometry/BoundingBox.h"

PrimitiveList
  ::~PrimitiveList(void)
{
  ;
} // end PrimitiveList::~PrimitiveList()

void PrimitiveList
  ::clear(void)
{
  Parent1::clear();
  mBoundingBox.setEmpty();
} // end PrimitiveList::clear()

void PrimitiveList
  ::push_back(const boost::shared_ptr<Primitive> &p)
{
  Parent1::push_back(p);

  BoundingBox b;
  p->getBoundingBox(b);
  mBoundingBox.addPoint(b[0]);
  mBoundingBox.addPoint(b[1]);
} // end PrimitiveList::push_back()

void PrimitiveList
  ::getBoundingBox(BoundingBox &b) const
{
  b = mBoundingBox;
} // end PrimitiveList::getBoundingBox()

bool PrimitiveList
  ::intersect(Ray &r, Intersection &inter) const
{
  bool result = false;
  for(Parent1::const_iterator p = Parent1::begin();
      p != Parent1::end();
      ++p)
  {
    result |= (*p)->intersect(r,inter);
  } // end for p

  return result;
} // end PrimitiveList::intersect()

bool PrimitiveList
  ::intersect(const Ray &r) const
{
  for(Parent1::const_iterator p = Parent1::begin();
      p != Parent1::end();
      ++p)
  {
    if((*p)->intersect(r)) return true;
  } // end for p

  return false;
} // end PrimitiveList::intersect()

void PrimitiveList
  ::finalize(void)
{
  Parent0::finalize();

  for(size_t i = 0; i != Parent1::size(); ++i)
  {
    (*this)[i]->setPrimitiveHandle(i);
  } // end for i
} // end PrimitiveList::finalize()

