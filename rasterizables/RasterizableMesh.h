/*! \file RasterizableMesh.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Mesh
 *         class which implements Rasterizable.
 */

#ifndef RASTERIZABLE_MESH_H
#define RASTERIZABLE_MESH_H

#include "Rasterizable.h"
#include "../surfaces/Mesh.h"

template<typename MeshParentType>
  class RasterizableMesh
    : public MeshParentType,
      public Rasterizable
{
  public:
    /*! \typedef Parent0
     *  \brief Shorthand.
     */
    typedef MeshParentType Parent0;

    /*! \typedef Parent1
     *  \brief Shorthand.
     */
    typedef Rasterizable Parent1;

    /*! Constructor takes a list of vertices and a list of
     *  triangles.
     *  \param vertices A list of vertex positions
     *  \param triangles A list of triangles.
     */
    inline RasterizableMesh(const std::vector<Point> &vertices,
                            const std::vector<typename Parent0::Triangle> &triangles);

    /*! Constructor takes a list of vertices and a list of
     *  triangles.
     *  \param points A list of vertex positions.
     *  \param parametrics A list of parametric vertex positions.
     *  \param triangles A list of triangles.
     */
    inline RasterizableMesh(const std::vector<Point> &points,
                            const std::vector<ParametricCoordinates> &parametrics,
                            const std::vector<typename Parent0::Triangle> &triangles);

    /*! Constructor takes a list of vertices and a list of
     *  triangles.
     *  \param points A list of vertex positions.
     *  \param parametrics A list of parametric vertex positions.
     *  \param normals A list of vertex normals.
     *  \param triangles A list of triangles.
     */
    inline RasterizableMesh(const std::vector<Point> &points,
                            const std::vector<ParametricCoordinates> &parametrics,
                            const std::vector<Normal> &normals,
                            const std::vector<typename Parent0::Triangle> &triangles);

    /*! This method rasterizes this RasterizableMesh using
     *  OpenGL commands.
     */
    inline virtual void rasterize(void);
}; // end RasterizableMesh

#include "RasterizableMesh.inl"

#endif // RASTERIZABLE_MESH_H

