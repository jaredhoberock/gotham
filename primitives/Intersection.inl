/*! \file Intersection.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Intersection.h.
 */

#include "Intersection.h"

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  const typename IntersectionBase<P3,V3,P2,N3>::DifferentialGeometry &IntersectionBase<P3,V3,P2,N3>
    ::getDifferentialGeometry(void) const
{
  return mDifferentialGeometry;
} // end IntersectionBase::getDifferentialGeometry()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  typename IntersectionBase<P3,V3,P2,N3>::DifferentialGeometry &IntersectionBase<P3,V3,P2,N3>
    ::getDifferentialGeometry(void)
{
  return mDifferentialGeometry;
} // end IntersectionBase::getDifferentialGeometry()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  void IntersectionBase<P3,V3,P2,N3>
    ::setDifferentialGeometry(const DifferentialGeometry &dg)
{
  mDifferentialGeometry = dg;
} // end IntersectionBase::setDifferentialGeometry()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  PrimitiveHandle IntersectionBase<P3,V3,P2,N3>
    ::getPrimitive(void) const
{
  return mPrimitive;
} // end IntersectionBase::getPrimitive()

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  void IntersectionBase<P3,V3,P2,N3>
    ::setPrimitive(const PrimitiveHandle p)
{
  mPrimitive = p;
} // end IntersectionBase::setPrimitive()

