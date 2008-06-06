/*! \file SmallMesh.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Mesh
 *         which is small enough not to use a
 *         kd-tree for ray intersection.
 */

#ifndef SMALL_MESH_H
#define SMALL_MESH_H

#include "Mesh.h"

class SmallMesh
  : public Mesh
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Mesh Parent;

    /*! Constructor takes a list of vertices and a list of triangles.
     *  \param vertices A list of vertex positions.
     *  \param triangles A list of triangles.
     */
    SmallMesh(const std::vector<Point> &vertices,
              const std::vector<Triangle> &triangles);

    /*! Constructor takes a list of positions, parametric positions, and a list of Triangles.
     *  \param vertices A list of vertex positions.
     *  \param parametrics A list of parametric vertex positions.
     *  \param triangles A list of triangles.
     */
    SmallMesh(const std::vector<Point> &vertices,
              const std::vector<ParametricCoordinates> &parametrics,
              const std::vector<Triangle> &triangles);

    /*! Constructor takes a list of positions, parametric positions, and a list of Triangles.
     *  \param vertices A list of vertex positions.
     *  \param parametrics A list of parametric vertex positions.
     *  \param normals A list of vertex normal vectors.
     *  \param triangles A list of triangles.
     */
    SmallMesh(const std::vector<Point> &vertices,
              const std::vector<ParametricCoordinates> &parametrics,
              const std::vector<Normal> &normals,
              const std::vector<Triangle> &triangles);

    /*! This method computes the intersection between the given Ray and this SmallMesh.
     *  If an intersection exists, the 'time' of intersection is returned.
     *  \param r The Ray to intersect.
     *  \param t If an intersection exists, the parametric value of the earliest
     *           non-negative intersection is returned here.
     *  \param dg If an intersection exists, a description of this SmallMesh's differential geometry at the intersection is returned here.
     *  \return true if r intersects this SmallMesh; false, otherwise.
     *  \note This method must be implemented in a child class.
     */
    virtual bool intersect(const Ray &r,
                           float &t,
                           DifferentialGeometry &dg) const;

    /*! This method computes whether or not an intersection between the given Ray and this SmallMesh exists.
     *  \param r The Ray to intersect.
     *  \return true if r intersects this Mesh; false, otherwise.
     */
    virtual bool intersect(const Ray &r) const;
}; // end SmallMesh

#endif // SMALL_MESH_H

