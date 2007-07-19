/*! \file MeshRasterizer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Rasterizer
 *         class for rasterizing Meshes.
 */

#ifndef MESH_RASTERIZER_H
#define MESH_RASTERIZER_H

#include "Rasterizer.h"
#include "../surfaces/Mesh.h"

class MeshRasterizer
  : public Rasterizer<Mesh>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Rasterizer<Mesh> Parent;

    MeshRasterizer(boost::shared_ptr<Mesh> mesh);
    virtual void operator()(void);
}; // end MeshRasterizer

#endif // MESH_RASTERIZER_H

