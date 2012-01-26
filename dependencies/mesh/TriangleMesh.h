/*! \file TriangleMesh.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class abstracting
 *         a mesh of triangles.
 */

#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

#include <vector>
#include <cstddef>

template<typename P3D, typename P2D, typename N3D = P3D>
  class TriangleMesh
{
  public:
    class Triangle
    {
      public:
        inline Triangle(void){}

        inline Triangle(const unsigned int v0,
                        const unsigned int v1,
                        const unsigned int v2)
        {
          mVertices[0] = v0;
          mVertices[1] = v1;
          mVertices[2] = v2;
        } // end Triangle()

        inline const unsigned int &operator[](const size_t i) const {return mVertices[i];}
        inline unsigned int &operator[](const size_t i) {return mVertices[i];}

      private:
        unsigned int mVertices[3];
    }; // end Triangle

    /*! Null constructor does nothing.
     */
    inline TriangleMesh(void);

    /*! Constructor accepts a list of points and Triangles.
     *  \param points The position of each vertex.
     *  \param triangles A list of Triangles.
     */
    inline TriangleMesh(const std::vector<P3D> &points,
                        const std::vector<Triangle> &triangles);

    /*! Constructor accepts a list of points, parametrics, and Triangles.
     *  \param points The position of each vertex.
     *  \param parametrics The parametric position of each vertex.
     *  \param triangles A list of Triangles.
     */
    inline TriangleMesh(const std::vector<P3D> &points,
                        const std::vector<P2D> &parametrics,
                        const std::vector<Triangle> &triangles);

    /*! Constructor accepts a list of points, parametrics, normals, and Triangles.
     *  \param points The position of each vertex.
     *  \param parametrics The parametric position of each vertex.
     *  \param normals The normal vector of each vertex.
     *  \param triangles A list of Triangles.
     */
    inline TriangleMesh(const std::vector<P3D> &points,
                        const std::vector<P2D> &parametrics,
                        const std::vector<N3D> &normals,
                        const std::vector<Triangle> &triangles);

    typedef std::vector<Triangle> TriangleList;
    typedef std::vector<P3D> PointList;
    typedef std::vector<P2D> ParametricList;
    typedef std::vector<N3D> NormalList;

    /*! This method returns mTriangles.
     *  \return mTriangles
     */
    const TriangleList &getTriangles(void) const;

    /*! This method returns mPositions.
     *  \return mPositions
     */
    const PointList &getPoints(void) const;

    /*! This method returns mNormals.
     *  \return mNormals
     */
    const NormalList &getNormals(void) const;

    /*! This method computes the surface area of this Mesh.
     *  \param t The Triangle of interest.
     *  \return The surface area of t.
     */
    float computeSurfaceArea(const Triangle &t) const;

  protected:
    PointList mPoints;
    ParametricList mParametrics;
    NormalList mNormals;
    TriangleList mTriangles;
}; // end TriangleMesh

#include "TriangleMesh.inl"

#endif // TRIANGLE_MESH_H

