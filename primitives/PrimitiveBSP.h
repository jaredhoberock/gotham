/*! \file PrimitiveBSP.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a
 *         PrimitiveList which also
 *         maintains a bsp tree for accellerated
 *         ray intersection.
 */

#ifndef PRIMITIVE_BSP_H
#define PRIMITIVE_BSP_H

#include "PrimitiveList.h"
#include <rayCaster/bsp.h>

class PrimitiveBSP
  : public PrimitiveList,
    protected bspTree<const Primitive*,Point>
{
  public:
    /*! \typedef Parent0
     *  \brief Shorthand.
     */
    typedef PrimitiveList Parent0;

    /*! \typedef Parent1
     *  \brief Shorthand.
     */
    typedef bspTree<const Primitive*,Point> Parent1;

    /*! Null constructor calls the Parents.
     */
    inline PrimitiveBSP(void);

    /*! This method finalizes this PrimitiveBSP.
     */
    inline virtual void finalize(void);

    /*! This method intersects a Ray against this PrimitiveBSP.
     *  \param r The Ray of interest.
     *  \param inter If an intersection exists, information regarding the
     *         geometry of the Intersection is returned here.
     *  \return true if an Intersection exists; false, otherwise.
     */
    inline virtual bool intersect(Ray &r, Intersection &inter) const;

    /*! This method intersects a Ray against this PrimitiveBSP.
     *  \param r The Ray of interest.
     *  \return true if an intersection exists; false, otherwise.
     */
    inline virtual bool intersect(const Ray &r) const;

  protected:
    /*! \class Bounder
     *  \brief A functor for bounding a Primitive.
     *  XXX DESIGN consider placing these in Primitive
     */
    struct Bounder
    {
      /*! operator()() method returns the bounds of a Primitive given
       *  the axis of interest.
       *  \param axis The dimension of interest.
       *  \param min Whether to return the minimum or maximum bound.
       *  \param p A const pointer to the Primitive of interest.
       *  \return A minimal or maximal bound on p on the specified axis.
       */
      inline float operator()(unsigned int axis, bool min, const Primitive *p) const;
    }; // end struct Bounder

    /*! \struct Intersector
     *  \brief A functor for intersecting a Ray with Primitives.
     *  XXX DESIGN consider placing these in Primitive
     */
    struct Intersector
    {
      /*! operator()() method performs ray intersection with a list of Primitives.
       *  \param anchor The anchor of the Ray to intersect.
       *  \param dir The direction of the Ray to intersect.
       *  \param begin The beginning of the list of Primitives to intersect against.
       *  \param end The end of the list of Primitives to intersect against.
       *  \param minT The parametric Ray value as the Ray enters the current BSP cell.
       *              No intersections will be counted before this point.
       *  \param maxT The parametric Ray value as the Ray leaves the current BSP cell.
       *              No intersections will be counted beyond this point.
       *  \return true if the Ray hits a Primitive in the list; false, otherwise.  If so, mT is set to the parametric Ray value at the
       *          hit point, and mHitPrimitive points to the intersected Primitive.
       */
      inline bool operator()(const Point &anchor, const Point &dir,
                             const Primitive** begin,
                             const Primitive** end,
                             float minT, float maxT);

      /*! Primitive::Intersection record of the hit Primitive.
       */
      Intersection mIntersection;

      /*! The Ray parameter value at the point of intersection.
       */
      float mHitTime;
    }; // end struct PrimitiveIntersector

    /*! \struct Shadower
     *  \brief A functor for intersecting a shadow Ray with Primitives.
     *  XXX DESIGN consider placing these in Primitive
     */
    struct Shadower
    {
      /*! operator()() method performs shadow ray intersection with a list of Primitives.
       *  \param anchor The anchor of the Ray to intersect.
       *  \param dir The direction of the Ray to intersect.
       *  \param begin The beginning of the list of Primitives to intersect against.
       *  \param end The end of the list of Primitives to intersect against.
       *  \param minT The parametric Ray value as the Ray enters the current BSP cell.
       *              No intersections will be counted before this point.
       *  \param maxT The parametric Ray value as the Ray leaves the current BSP cell.
       *              No intersections will be counted beyond this point.
       *  \return true if the Ray hits a Primitive in the list; false, otherwise.  If so, mT is set to the parametric Ray value at the
       *          hit point, and mHitPrimitive points to the intersected Primitive.
       */
      inline bool operator()(const Point &anchor, const Point &dir,
                             const Primitive** begin,
                             const Primitive** end,
                             float minT, float maxT);
    }; // end Shadower
}; // end PrimitiveBSP

#include "PrimitiveBSP.inl"

#endif // PRIMITIVE_BSP_H

