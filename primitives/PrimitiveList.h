/*! \file PrimitiveList.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a list of
 *         Primitives.
 */

#ifndef PRIMITIVE_LIST_H
#define PRIMITIVE_LIST_H

#include <vector>
#include <boost/shared_ptr.hpp>
#include "Primitive.h"
#include "../geometry/BoundingBox.h"

class Ray;

template<typename PrimitiveType = Primitive>
  class PrimitiveList
    : public Primitive,
      protected std::vector<boost::shared_ptr<PrimitiveType> >
{
  public:
    typedef Primitive Parent0;

    inline virtual void push_back(boost::shared_ptr<PrimitiveType> &p);
    inline virtual void clear(void);
    inline virtual void getBoundingBox(BoundingBox &b) const;
    inline virtual bool intersect(Ray &r, Intersection &inter) const;
    inline virtual bool intersect(const Ray &r) const;

  private:
    typedef std::vector<boost::shared_ptr<PrimitiveType> > Parent1;
    BoundingBox mBoundingBox;
}; // end PrimitiveList

#include "PrimitiveList.inl"

#endif // PRIMITIVE_LIST_H

