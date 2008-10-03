/*! \file Mesh.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a surface representing a mesh.
 */

#ifndef MESH_H
#define MESH_H

#include <mesh/TriangleMesh.h>
#include "../include/detail/Point.h"
#include "../include/detail/Normal.h"
#include "../include/detail/ParametricCoordinates.h"
#include "../include/detail/DifferentialGeometry.h"
#include "../geometry/Ray.h"
#include "../geometry/BoundingBox.h"
#include <rayCaster/bsp.h>
#include <aliastable/AliasTable.h>
#include "Surface.h"

class Mesh
  : public TriangleMesh<Point, ParametricCoordinates, Normal>,
    public Surface
{
  public:
    /*! \typedef Parent0
     *  \brief Shorthand.
     */
    typedef TriangleMesh<Point, ParametricCoordinates, Normal> Parent0;

    /*! \typedef Parent1
     *  \brief Shorthand.
     */
    typedef Surface Parent1;

    /*! This structure encapsulates the data needed to do
     *  fast Wald-Bikker style ray-triangle intersection.
     */
    struct WaldBikkerData
    {
      gpcpu::float3 mN;
      gpcpu::float2 mBn;
      gpcpu::float2 mCn;

      // we need two bits for the dominant axis
      // we need not store u and v
      unsigned int mDominantAxis:3;
      unsigned int mUAxis:3;
      unsigned int mVAxis:3;
    }; // end WaldBikkerData

    /*! Constructor takes a list of vertices and a list of triangles.
     *  \param vertices A list of vertex positions.
     *  \param triangles A list of triangles.
     */
    Mesh(const std::vector<Point> &vertices,
         const std::vector<Triangle> &triangles);

    /*! Constructor takes a list of positions, parametric positions, and a list of Triangles.
     *  \param vertices A list of vertex positions.
     *  \param parametrics A list of parametric vertex positions.
     *  \param triangles A list of triangles.
     */
    Mesh(const std::vector<Point> &vertices,
         const std::vector<ParametricCoordinates> &parametrics,
         const std::vector<Triangle> &triangles);

    /*! Constructor takes a list of positions, parametric positions, normals, and a list of Triangles.
     *  \param vertices A list of vertex positions.
     *  \param parametrics A list of parametric vertex positions.
     *  \param normals A list of vertex normals.
     *  \param triangles A list of triangles.
     */
    Mesh(const std::vector<Point> &vertices,
         const std::vector<ParametricCoordinates> &parametrics,
         const std::vector<Normal> &normals,
         const std::vector<Triangle> &triangles);

    /*! Null destructor does nothing.
     */
    virtual ~Mesh(void);

    /*! This method returns a BoundingBox bounding the vertices of this Mesh.
     *  \param b The BoundingBox bounding the vertices of this Mesh is returned here.
     */
    virtual void getBoundingBox(BoundingBox &b) const;

    /*! This method computes the intersection between the given Ray and this Mesh.
     *  If an intersection exists, the 'time' of intersection is returned.
     *  \param r The Ray to intersect.
     *  \param t If an intersection exists, the parametric value of the earliest
     *           non-negative intersection is returned here.
     *  \param dg If an intersection exists, a description of this Mesh's differential geometry at the intersection is returned here.
     *  \return true if r intersects this Mesh; false, otherwise.
     *  \note This method must be implemented in a child class.
     */
    virtual bool intersect(const Ray &r,
                           float &t,
                           DifferentialGeometry &dg) const;

    /*! This method computes whether or not an intersection between the given Ray and this Mesh exists.
     *  \param r The Ray to intersect.
     *  \return true if r intersects this Mesh; false, otherwise.
     */
    virtual bool intersect(const Ray &r) const;

    /*! \class TriangleBounder
     *  \brief Functor for providing bounds of a triangle's vertex positions.
     */
    struct TriangleBounder
    {
      /*! operator()() method performs triangle bounding.
       *  \param axis Which dimension to bound.
       *  \param min Whether or not to return the min or max bound.
       *  \param t The Triangle to bound.
       *  \return The min or max vertex coordinate of the given triangle's vertices.
       */
      float operator()(unsigned int axis, bool min, const Triangle *t);

      /*! A pointer to the Mesh containing the triangles we're bounding.
       */
      const Mesh *mMesh;
    }; // end struct TriangleBounder

    /*! \class TriangleIntersector
     *  \brief Functor for intersecting a ray against a triangular Face.
     */
    struct TriangleIntersector
    {
      /*! This method clears the TriangleIntersector structure and initializes it for intersection.
       */
      void init(void);

      /*! operator()() method performs ray intersection with a triangle.
       *  \param anchor The anchor of the Ray.
       *  \param dir The direction of the Ray.
       *  \param begin Pointer to the first triangle to intersect against.
       *  \param end Pointer to the end of the triangle list to intersect against.
       *  \param minT The parametric value along the Ray where it enters the bsp cell.
       *  \param maxT The parametric value along the Ray where it exits the bsp cell.
       *  \return true if the ray hits a triangle before maxT; false, otherwise.
       */
      inline bool operator()(const Point &anchor, const Point &dir,
                             const Triangle **begin, const Triangle **end,
                             float minT, float maxT);

      /*! The parametric ray value at the last hit.
       */
      float mT;

      /*! The barycentric coordinates at the hit location.
       */
      ParametricCoordinates mBarycentricCoordinates;

      /*! Pointer to the hit Triangle.
       */
      const Triangle *mHitFace;

      /*! A pointer to the Mesh containing the triangles we're intersecting against.
       */
      const Mesh *mMesh;
    }; // end class TriangleIntersector

    /*! \class TriangleShadower
     *  \brief Functor for intersecting a shadow ray against a triangular Face.
     */
    struct TriangleShadower
    {
      /*! operator()() method performs ray intersection with a triangle.
       *  \param anchor The anchor of the Ray.
       *  \param dir The direction of the Ray.
       *  \param begin Pointer to the first triangle to intersect against.
       *  \param end Pointer to the end of the triangle list to intersect against.
       *  \param minT The parametric value along the Ray where it enters the bsp cell.
       *  \param maxT The parametric value along the Ray where it exits the bsp cell.
       *  \return true if the ray hits a triangle before maxT; false, otherwise.
       */
      inline bool operator()(const Point &anchor, const Point &dir,
                             const Triangle **begin, const Triangle **end,
                             float minT, float maxT);

      /*! A pointer to the Mesh containing the triangles we're shadowing against.
       */
      const Mesh *mMesh;
    }; // end TriangleShadower

    /*! This method samples a point from a uniform distribution over the
     *  surface area of this Mesh.
     *  \param u1 A number in [0,1).
     *  \param u2 A second number in [0,1).
     *  \param dg The DifferentialGeometry at the sampled Point is
     *            returned here.
     *  \param pdf The value of the surface area pdf at p is returned here.
     */
    virtual void sampleSurfaceArea(const float u1,
                                   const float u2,
                                   const float u3,
                                   DifferentialGeometry &dg,
                                   float &pdf) const;

    /*! This method gets the DifferentialGeometry at the given
     *  barycentric coordinates in a Triangle of this Mesh.
     *  \param tri The Triangle of interest.
     *  \param p The 3D Point on tri.
     *  \param b1 The first barycentric coordinate.
     *  \param b2 The second barycentric coordinate.
     *  \dg The DifferentialGeometry at (b1,b2) inside tri is
     *      returned here.
     */
    void getDifferentialGeometry(const Triangle &tri,
                                 const Point &p,
                                 const float b1,
                                 const float b2,
                                 DifferentialGeometry &dg) const;

    /*! This method returns the ParametricCoordinates for the given Triangle.
     *  \param t The Triangle of interest.
     *  \param uv0 The ParametricCoordinates of the first vertex are returned here.
     *  \param uv1 The ParametricCoordinates of the second vertex are returned here.
     *  \param uv2 The ParametricCoordinates of the third vertex are returned here.
     */
    inline void getParametricCoordinates(const Triangle &tri,
                                         ParametricCoordinates &uv0,
                                         ParametricCoordinates &uv1,
                                         ParametricCoordinates &uv2) const;

    /*! This method evaluates the surface area pdf at a
     *  DifferentialGeometry of interest.
     *  \param dg The DifferentialGeometry describing the Point of interest.
     *  \return The value of the surface area pdf at dg.
     */
    virtual float evaluateSurfaceAreaPdf(const DifferentialGeometry &dg) const;

    /*! This method returns the surface area of this Mesh.
     *  \return mSurfaceArea
     */
    virtual float getSurfaceArea(void) const;

    /*! This method returns the surface area of this Mesh.
     *  \return mOneOverSurfaceArea
     */
    virtual float getInverseSurfaceArea(void) const;

    /*! This static method intersects a Triangle in a Mesh.
     *  \param o The origin of the Ray to test for intersection.
     *  \param d The direction of the Ray to test for intersection.
     *  \param f The Triangle to intersect.
     *  \param m The Mesh owning f.
     *  \param t If an intersection occurs, the intersection point is returned here.
     *  \param b1 The first barycentric coordinate of the intersection is returned here.
     *  \param b2 The second barycentric coordinate of the intersection is returned here.
     *  \return true if r intersects f; false, otherwise.
     */
    inline static bool intersect(const Point &o,
                                 const Vector &dir,
                                 const Triangle &f,
                                 const Mesh &m,
                                 float &t,
                                 float &b1,
                                 float &b2);

    /*! This method creates WaldBikkerData from a Triangle of interest.
     *  \param tri The Triangle of interest.
     *  \param data tri's WaldBikkerData is returned here.
     */
    void getWaldBikkerData(const Triangle &tri,
                           WaldBikkerData &data) const;

  protected:
    /*! This method builds the acceleration structure.
     */
    inline void buildTree(void);

    inline static bool intersectWaldBikker(const Point &o,
                                           const Vector &dir,
                                           const Triangle &f,
                                           const Mesh &m,
                                           const float minT,
                                           const float maxT,
                                           float &t,
                                           float &b1,
                                           float &b2);
                                           

    /*! \typedef KDTree
     *  \brief Shorthand.
     */
    typedef bspTree<const Triangle*,Point> KDTree;

    KDTree mTree;

    /*! An AliasTable for picking points uniformly from the surface
     *  area of this Mesh.
     */
    typedef AliasTable<const Triangle*,float> TriangleTable;
    TriangleTable mTriangleTable;

    /*! This method builds mTriangleTable.
     */
    virtual void buildTriangleTable(void);

    /*! This method computes the surface area of this Mesh.
     *  \return The surface area of this Mesh.
     */
    float computeSurfaceArea(void) const;

    /*! The surface area of this Mesh.
     */
    float mSurfaceArea;

    /*! One over the surface area of this Mesh.
     */
    float mOneOverSurfaceArea;

    std::vector<WaldBikkerData> mWaldBikkerTriangleData;

    void buildWaldBikkerData(void);

    /*! Whether or not to interpolate normals.
     *  If so, then mNormals is interpreted as a per-vertex list of Normals.
     *  Otherwise, it is per-triangle.
     */
    bool mInterpolateNormals;

    /*! This method creates per-face normals for each triangle.
     */
    void createTriangleNormals(void);
}; // end Mesh

#include "Mesh.inl"

#endif // MESH_H

