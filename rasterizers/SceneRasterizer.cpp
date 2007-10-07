/*! \file SceneRasterizer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SceneRasterizer class.
 */

#include "SceneRasterizer.h"
#include "../primitives/SurfacePrimitive.h"
#include "../surfaces/Mesh.h"
#include "../primitives/SurfacePrimitiveList.h"

SceneRasterizer
  ::SceneRasterizer(void)
    :Parent()
{
  ;
} // end SceneRasterizer::SceneRasterizer()

SceneRasterizer
  ::SceneRasterizer(boost::shared_ptr<Scene> p)
    :Parent(p)
{
  setPrimitive(p);
} // end SceneRasterizer::SceneRasterizer()

void SceneRasterizer
  ::setPrimitive(boost::shared_ptr<Scene> p)
{
  // call the Parent
  Parent::setPrimitive(p);

  // clear the old rasterizers
  mMeshRasterizers.clear();

  // for each shape we find that we know, rasterize it
  const SurfacePrimitive *prim = dynamic_cast<SurfacePrimitive*>(mPrimitive.get());
  if(prim != 0)
  {
    const Mesh *m = dynamic_cast<const Mesh*>(prim->getSurface());
    if(m != 0)
    {
      // cast to Mesh
      boost::shared_ptr<Surface> surf;
      prim->getSurface(surf);
      boost::shared_ptr<Mesh> mesh
        = boost::dynamic_pointer_cast<Mesh, Surface>(surf);

      // add a MeshRasterizer
      mMeshRasterizers.push_back(MeshRasterizer(mesh));
    } // end if
  } // end if
  else
  {
    const SurfacePrimitiveList *list
      = dynamic_cast<SurfacePrimitiveList*>(mPrimitive.get());
    if(list != 0)
    {
      boost::shared_ptr<SurfacePrimitiveList> primList
        = boost::dynamic_pointer_cast<SurfacePrimitiveList, Primitive>(primList);
      mListRasterizers.push_back(SurfacePrimitiveListRasterizer(primList));
    } // end if
  } // end else
} // end SceneRasterizer::setScene()

void SceneRasterizer
  ::operator()(void)
{
  for(unsigned int i = 0;
      i != mMeshRasterizers.size();
      ++i)
  {
    mMeshRasterizers[i]();
  } // end for r
} // end SceneRasterizer::operator()()

