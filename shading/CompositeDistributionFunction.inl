/*! \file CompositeDistributionFunction.inl
 *  \author Jared Hoberock
 *  \brief Inline file for CompositeDistributionFunction.h.
 */

#include "CompositeDistributionFunction.h"

CompositeDistributionFunction
  ::CompositeDistributionFunction(void)
    :Parent0(),Parent1(),mSize(0)
{
  ;
} // end CompositeDistributionFunction::CompositeDistributionFunction

CompositeDistributionFunction &CompositeDistributionFunction
  ::operator+=(ScatteringDistributionFunction *rhs)
{
  if(mSize < Parent1::static_size)
  {
    (*this)[mSize] = rhs;
    ++mSize;
  } // end if

  return *this;
} // end CompositeDistributionFunction::operator+=()

CompositeDistributionFunction &CompositeDistributionFunction
  ::operator+=(ScatteringDistributionFunction &rhs)
{
  return (*this) += &rhs;
} // end CompositeDistributionFunction::operator+=()

