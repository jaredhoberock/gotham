/*! \file RasterizablePrimitiveList.inl
 *  \author Jared Hoberock
 *  \brief Inline file for RasterizablePrimitiveList.h.
 */

#include "RasterizablePrimitiveList.h"

template<typename PrimitiveListParentType>
  void RasterizablePrimitiveList<PrimitiveListParentType>
    ::rasterize(void)
{
  for(typename Parent0::iterator prim = Parent0::begin();
      prim != Parent0::end();
      ++prim)
  {
    Rasterizable *r = dynamic_cast<Rasterizable*>((*prim).get());
    if(r != 0)
    {
      r->rasterize();
    } // end if
  } // end for prim
} // end RasterizablePrimitiveList::rasterize()
