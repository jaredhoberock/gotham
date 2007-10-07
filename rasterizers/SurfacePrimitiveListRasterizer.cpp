/*! \file SurfacePrimitiveListRasterizer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SurfacePrimitiveListRasterizer class.
 */

#include "SurfacePrimitiveListRasterizer.h"

SurfacePrimitiveListRasterizer
  ::SurfacePrimitiveListRasterizer(void)
    :Parent()
{
  ;
} // end SurfacePrimitiveListRasterizer::SurfacePrimitiveListRasterizer()

SurfacePrimitiveListRasterizer
  ::SurfacePrimitiveListRasterizer(boost::shared_ptr<SurfacePrimitiveList> p)
    :Parent(p)
{
  setPrimitive(p);
} // end SurfacePrimitiveListRasterizer::SurfacePrimitiveListRasterizer()

void SurfacePrimitiveListRasterizer
  ::setPrimitive(boost::shared_ptr<SurfacePrimitiveList> p)
{
  // call the Parent
  Parent::setPrimitive(p);

  // clear the old rasterizers
  mMeshRasterizers.clear();

  // for each shape we find that we know, rasterize it
  for(SurfacePrimitiveList::const_iterator prim = mPrimitive->begin();
      prim != mPrimitive->end();
      ++prim)
  {
    const Mesh *m = dynamic_cast<const Mesh*>((**prim).getSurface());
    if(m != 0)
    {
      // cast to Mesh
      boost::shared_ptr<Surface> surf;
      (**prim).getSurface(surf);
      boost::shared_ptr<Mesh> mesh
        = boost::dynamic_pointer_cast<Mesh, Surface>(surf);

      // add a MeshRasterizer
      mMeshRasterizers.push_back(MeshRasterizer(mesh));
    } // end if
  } // end for
} // end SurfacePrimitiveListRasterizer::setScene()

void SurfacePrimitiveListRasterizer
  ::operator()(void)
{
  for(unsigned int i = 0;
      i != mMeshRasterizers.size();
      ++i)
  {
    mMeshRasterizers[i]();
  } // end for r
} // end SurfacePrimitiveListRasterizer::operator()()

