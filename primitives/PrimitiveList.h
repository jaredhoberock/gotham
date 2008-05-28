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

class PrimitiveList
  : public Primitive,
    public std::vector<boost::shared_ptr<Primitive> >
{
  public:
    typedef Primitive Parent0;

    /*! Null destructor does nothing.
     */
    virtual ~PrimitiveList(void);

    virtual void push_back(const boost::shared_ptr<Primitive> &p);
    virtual void clear(void);
    virtual void getBoundingBox(BoundingBox &b) const;
    
    using Parent0::intersect;
    virtual bool intersect(Ray &r, Intersection &inter) const;
    virtual bool intersect(const Ray &r) const;

  private:
    typedef std::vector<boost::shared_ptr<Primitive> > Parent1;
    BoundingBox mBoundingBox;
}; // end PrimitiveList

#endif // PRIMITIVE_LIST_H

