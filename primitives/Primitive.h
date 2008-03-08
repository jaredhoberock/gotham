/*! \file Primitive.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class representing a primitive renderable element of a Scene.
 */

#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "exportPrimitives.h"

class BoundingBox;
class Ray;
class Intersection;
#include "../geometry/DifferentialGeometry.h"
#include "../geometry/Transform.h"
#include <spectrum/Spectrum.h>
#include <boost/functional/hash.hpp>

/*! \class Primitive
 *  \brief The Primitive base class bridges geometry processing and shading.  It provides an interface
 *         for performing Ray intersection and shading queries against it.
 */
class Primitive
{
  public:
    /*! \class Intersection
     *  \brief Intersection records information about a Ray's intersection with a Primitive.
     */
    class Intersection
    {
      public:
        /*! This method returns a const reference to mDifferentialGeometry.
         *  \return mDifferentialGeometry.
         */
        inline const DifferentialGeometry &getDifferentialGeometry(void) const;

        /*! This method returns a reference to mDifferentialGeometry.
         *  \return mDifferentialGeometry.
         */
        inline DifferentialGeometry &getDifferentialGeometry(void);

        /*! This method sets mDifferentialGeometry.
         *  \param dg Sets mDifferentialGeometry.
         */
        inline void setDifferentialGeometry(const DifferentialGeometry &dg);

        /*! This method returns mPrimitive.
         *  \return mPrimitive
         */
        inline const Primitive *getPrimitive(void) const;

        /*! This method sets mPrimitive.
         *  \param p Sets mPrimitive.
         */
        inline void setPrimitive(const Primitive *p);

      protected:
        /*! An Intersection keeps a record of the DifferentialGeometry of the surface hit.
         */
        DifferentialGeometry mDifferentialGeometry;

        /*! An Intersection keeps a pointer to the Primitive hit. */
        const Primitive *mPrimitive;
    }; // end class Intersection

    /*! \brief Null constructor does nothing.
     */
    inline Primitive(void);

    /*! Null destructor does nothing.
     */
    inline virtual ~Primitive(void);

    /*! A Primitive must be able to return a BoundingBox bounding it in world space.
     *  \param b A BoundingBox bounding this Primitive is returned here.
     *  \note Must be implemented in a child class.
     */
    virtual void getBoundingBox(BoundingBox &b) const = 0;

    /*! If a Primitive is intersectable, it must provide a way to query for a Ray intersection.
     *  \param r The Ray to intersect.  If an intersection exists, r's legal parametric interval is updated accordingly.
     *           This is the only part of r that will be changed.
     *  \param inter If an Intersection exists, information regarding the first intersection along r is returned here.
     *  \return true if an Intersection exists; false, otherwise.
     *  \note The default implementation of Primitive::intersect() always trivially returns false.
     */
    inline virtual bool intersect(Ray &r, Intersection &inter) const;

    /*! This method provides a SIMD path for intersect(). It intersects more than
     *  one Ray against this Primitive en masse.
     *  \param rays A list of Rays to intersect.
     *  \param rays A list of Rays to intersect.
     *  \param intersections Af an intersection for a Ray exists, a Primitive::Intersection record storing information about the first
     *         intersection encountered is returned here.
     *  \param stencil If a Ray hits something, this is set to true.
     *  \param n The length of lists rays, intersections, and stencil.
     *  \note The default implementation of this method calls the single Ray
     *        version of intersect() repeatedly.
     */
    inline virtual void intersect(Ray *rays,
                                  Intersection *intersections,
                                  int *stencil,
                                  const size_t n) const;

    /*! If a Primitive is intersectable, it must provide a way to query for a Ray intersection.
     *  \param r The Ray to intersect.
     *  \return true if an intersection between this Primitive and r exists; false, otherwise.
     *  \note The default implementation of Primitive::intersect() always trivially returns false.
     */
    inline virtual bool intersect(const Ray &r) const;

    /*! This method sets this Primitive's name.
     *  \param name Sets mName.
     */
    inline void setName(const std::string &name);

    /*! This method returns the name of this Primitive.
     *  \return mName.
     */
    inline const std::string &getName(void) const;

    /*! This method returns mNameHash.
     *  \return mNameHash.
     */
    inline size_t getNameHash(void)const;

    /*! This method provides a means of doing last minute
     *  tasks immediately prior to beginning a render.
     *  \note The default implementation does nothing.
     */
    inline virtual void finalize(void);

  protected:
    /*! A name, possibly the null string, associated with this Primitive.
     */
    std::string mName;

    /*! The hash of mName.
     */
    size_t mNameHash;

    static boost::hash<std::string> mStringHasher;
}; // end class Primitive

#include "Primitive.inl"

#endif // PRIMITIVE_H

