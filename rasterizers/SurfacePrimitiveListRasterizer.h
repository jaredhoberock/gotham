/*! \file SurfacePrimitiveListRasterizer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Rasterizer
 *         for SurfacePrimitiveLists.
 */

#ifndef SURFACE_PRIMITIVE_LIST_RASTERIZER_H
#define SURFACE_PRIMITIVE_LIST_RASTERIZER_H

#include "../primitives/SurfacePrimitiveList.h"
#include "Rasterizer.h"
#include "MeshRasterizer.h"

class SurfacePrimitiveListRasterizer
  : public Rasterizer<SurfacePrimitiveList>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Rasterizer<SurfacePrimitiveList> Parent;

    /*! Null constructor does nothing.
     */
    SurfacePrimitiveListRasterizer(void);

    SurfacePrimitiveListRasterizer(boost::shared_ptr<SurfacePrimitiveList> s);
    virtual void operator()(void);
    virtual void setPrimitive(boost::shared_ptr<SurfacePrimitiveList> s);

  protected:
    std::vector<MeshRasterizer> mMeshRasterizers;
}; // end SurfacePrimitiveListRasterizer

#endif // SURFACE_PRIMITIVE_LIST_RASTERIZER_H

