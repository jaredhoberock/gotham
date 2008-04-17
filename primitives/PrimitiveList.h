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
    inline virtual ~PrimitiveList(void);

    inline virtual void push_back(const boost::shared_ptr<Primitive> &p);
    inline virtual void clear(void);
    inline virtual void getBoundingBox(BoundingBox &b) const;
    inline virtual bool intersect(Ray &r, Intersection &inter) const;
    inline virtual bool intersect(const Ray &r) const;

    /*! This method calls setPrimitiveHandle on each element of this
     *  PrimitiveList.
     */
    inline virtual void finalize(void);

  private:
    typedef std::vector<boost::shared_ptr<Primitive> > Parent1;
    BoundingBox mBoundingBox;
}; // end PrimitiveList

#include "PrimitiveList.inl"

#endif // PRIMITIVE_LIST_H

