/*! \file Transform.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Transform.h.
 */

#include "Transform.h"

Transform
  ::Transform(void):Parent()
{
  ;
} // end Transform::Transform()

Transform
  ::Transform(const gpcpu::float4x4 &xfrm)
    :Parent(xfrm,xfrm.inverse())
{
  ;
} // end Transform::Transform()

Transform
  ::Transform(const gpcpu::float4x4 &xfrm,
              const gpcpu::float4x4 &inv)
    :Parent(xfrm, inv)
{
  ;
} // end Transform::Transform

Transform Transform
  ::identity(void)
{
  gpcpu::float4x4 i = gpcpu::float4x4::identity();
  return Transform(i,i);
} // end Transform::identity()

void Transform
  ::getInverse(Transform &inv) const
{
  inv.set(Parent::second, Parent::first);
} // end Transform::getInverse()

void Transform
  ::set(const gpcpu::float4x4 &xfrm,
        const gpcpu::float4x4 &inv)
{
  Parent::first = xfrm;
  Parent::second = inv;
} // end Transform::set()

