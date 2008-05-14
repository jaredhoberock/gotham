/*! \file Intersection.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class
 *         bridging DifferentialGeometry at a
 *         Ray intersection point with the Primitive
 *         at that point.
 */

#pragma once

#include "../include/DifferentialGeometry.h"
#include "PrimitiveHandle.h"

/*! \class IntersectionBase
 *  \brief IntersectionBase records information about a Ray's IntersectionBase with a Primitive.
 *  \note This is a template to allow us to substitute CUDA vectors for the
 *        vector types.  Thanks, CUDA!
 */
template<typename P3 = Point,
         typename V3 = Vector,
         typename P2 = ParametricCoordinates,
         typename N3 = Normal>
  class IntersectionBase
{
  public:
    typedef DifferentialGeometryBase<P3,V3,P2,N3> DifferentialGeometry;

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
    inline PrimitiveHandle getPrimitive(void) const;

    /*! This method sets mPrimitive.
     *  \param p Sets mPrimitive.
     */
    inline void setPrimitive(const PrimitiveHandle p);

  protected:
    /*! An IntersectionBase keeps a record of the DifferentialGeometry of the surface hit.
     */
    DifferentialGeometry mDifferentialGeometry;

    /*! An IntersectionBase keeps a handle to the Primitive hit. */
    PrimitiveHandle mPrimitive;

    float mPaddingForCUDA;
}; // end class IntersectionBase

typedef IntersectionBase<> Intersection;

#include "Intersection.inl"

