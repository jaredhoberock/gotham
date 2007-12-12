/*! \file Ray.h
 *  \author Jared Hoberock
 *  \brief Defines the interface for a class abstracting a mathematical ray.
 */

#ifndef RAY_H
#define RAY_H

#include <gpcpu/Vector.h>
#include "Vector.h"
#include "Point.h"

/*! \class Ray
 *  \brief A Ray is a semi-infinite line, with an anchor and direction.
 *         A Ray's direction is not necessarily normalized.  The user can also
 *         specify a legal range of parameter values for Points on the Ray.  In
 *         this way, a Ray is more like a line segment.
 */
class Ray
{
  public:
    /*! \typedef Interval
     *  \brief Shorthand.
     */
    typedef gpcpu::float2 Interval;

    /*! Null constructor sets mInterval = [EPSILON, INFINITY].
     */
    inline Ray(void);

    /*! Constructor accepts an anchor and direction.
     *  \param a Sets mAnchor.
     *  \param d Sets mDirection.
     *  \note Sets mInterval = [EPSILON, INFINITY].
     */
    inline Ray(const Point &a, const Vector &d);

    /*! Constructor accepts two Points.
     *  \param p0 The origin.
     *  \param p1 The destination.
     */
    inline Ray(const Point &p0, const Point &p1);

    /*! Constructor accepts an anchor and direction.
     *  \param a Sets mAnchor.
     *  \param d Sets mDirection.
     *  \param mint Sets mInterval[0].
     *  \param maxt Sets mInterval[1].
     */
    inline Ray(const Point &a, const Vector3 &d, const float mint, const float maxt);

    /*! Constructor accepts an anchor and direction.
     *  \param a Sets mAnchor.
     *  \param d Sets mDirection.
     *  \param maxt Sets mInterval[1].
     *  \note mInterval[0] is set to EPSILON.
     */
    inline Ray(const Point &a, const Vector3 &d, const float maxt);

    /*! Null destructor does nothing.
     */
    inline virtual ~Ray(void);

    /*! This method returns a const reference to mAnchor.
     *  \return mAnchor
     */
    inline const Point &getAnchor(void) const;

    /*! This method sets this Ray's anchor.
     *  \param a Sets mAnchor.
     */
    inline void setAnchor(const Point &a);

    /*! This method returns a const reference to mDirection.
     *  \return mDirection
     */
    inline const Vector3 &getDirection(void) const;

    /*! This method returns a const reference to mInverseDirection.
     *  \return mInverseDirection
     */
    inline const Vector3 &getInverseDirection(void) const;

    /*! This method sets this Ray's direction.
     *  \param d Sets mDirection.
     */
    inline void setDirection(const Vector3 &d);

    /*! \brief This method evaluates the Ray equation, a + t*d.
     *  \param t The parametric value at which to evaluate the equation.
     *  \return evaluate(t)
     */
    inline Point operator()(const float t) const;

    /*! This method evaluates the Ray equation, a + t*d.
     *  \param t The parameteric value at which to evaluate the equation.
     *  \return mAnchor + t*mDirection
     */
    inline Point evaluate(const float t) const;

    /*! This method sets this Ray's anchor and direction by creating the Ray anchored at point p0
     *  that passes through p1.  The direction is set as p1 - p0, and NOT normalized.
     *  \param p0 Sets mAnchor.
     *  \param p1 Sets this Ray to pass through this Point.
     */
    inline void set(const Point &p0, const Point &p1);

    /*! This method returns a reference to mInterval.
     *  \return mInterval
     */
    inline Interval &getInterval(void);

    /*! This method returns a const reference to mInterval.
     *  \return mInterval.
     */
    inline const Interval &getInterval(void) const;

    /*! A very small value.
     */
    static const float RAY_EPSILON;

    /*! A very large value.
     */
    static const float RAY_INFINITY;

    /*! This method returns true if the given parameter value lies within mInterval.
     *  \param t The parameter value to test.
     *  \return mInterval[0] <= t && t <= mInterval[1].
     */
    inline virtual bool contains(const float t) const;

  protected:
    /*! A Ray has an anchor.
     */
    Point mAnchor;

    /*! A Ray has a (not necessarily normalized) direction.
     */
    Vector3 mDirection;

    /*! The inverse of each of the coordinates of mDirection.
     */
    Vector3 mInverseDirection;

    /*! A Ray has a legal Interval of parameter values which
     *  lie upon it.
     */
    Interval mInterval;
}; // end class Ray

#include "Ray.inl"

#endif // RAY_H

