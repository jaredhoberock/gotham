/*! \file SceneRasterizer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SceneRasterizer class.
 */

#include "SceneRasterizer.h"
#include "../primitives/SurfacePrimitive.h"
#include "../surfaces/Mesh.h"

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
  //// call the Parent
  //Parent::setPrimitive(p);

  //// clear the old rasterizers
  //mMeshRasterizers.clear();

  //// for each shape we find that we know, rasterize it
  //const SurfacePrimitive *prim = dynamic_cast<SurfacePrimitive*>(mPrimitive->getPrimitive().get());
  //if(prim != 0)
  //{
  //  const Mesh *m = dynamic_cast<const Mesh*>(prim->getSurface().get());
  //  if(m != 0)
  //  {
  //    // cast to Mesh
  //    boost::shared_ptr<Mesh> mesh
  //      = boost::dynamic_pointer_cast<Mesh, Surface>(prim->getSurface());

  //    // add a MeshRasterizer
  //    mMeshRasterizers.push_back(MeshRasterizer(mesh));
  //  } // end if
  //} // end if
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

