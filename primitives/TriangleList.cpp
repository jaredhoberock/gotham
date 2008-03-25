/*! \file TriangleList.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of TriangleList class.
 */

#include "TriangleList.h"
#include "../surfaces/Surface.h"
#include "SurfacePrimitive.h"
#include "../surfaces/Mesh.h"

void TriangleList
  ::push_back(const boost::shared_ptr<ListElement> &p)
{
  // XXX DESIGN this sucks
  const SurfacePrimitive *sp = dynamic_cast<const SurfacePrimitive*>(p.get());
  if(sp)
  {
    const Mesh *mesh = dynamic_cast<const Mesh *>(sp->getSurface());
    if(mesh)
    {
      Parent::push_back(p);
    } // end if
  } // end if
} // end TriangleList::push_back()

