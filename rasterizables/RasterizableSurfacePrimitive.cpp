/*! \file RasterizableSurfacePrimitive.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of RasterizableSurfacePrimitive class.
 */

#include "RasterizableSurfacePrimitive.h"

RasterizableSurfacePrimitive
  ::RasterizableSurfacePrimitive(boost::shared_ptr<Surface> s,
                                 const MaterialHandle &m)
    :Parent0(s,m),Parent1()
{
  ;
} // end RasterizableSurfacePrimitive::RasterizableSurfacePrimitive()

void RasterizableSurfacePrimitive
  ::rasterize(void)
{
  Rasterizable *r = dynamic_cast<Rasterizable*>(mSurface.get());
  if(r != 0)
  {
    r->rasterize();
  } // end if
  else
  {
    // XXX rasterize mSurface's bounding box here?
  } // end else
} // end RasterizableSurfacePrimitive::rasterize()

