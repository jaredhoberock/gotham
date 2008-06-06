/*! \file Mesh.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Mesh class.
 */

#include "Mesh.h"
#include <2dmapping/UnitSquareToTriangle.h>
#include <waldbikkerintersection/waldBikkerIntersection.h>

Mesh
  ::Mesh(const std::vector<Point> &vertices,
         const std::vector<Triangle> &triangles)
     :Parent0(vertices,triangles),Parent1(),mInterpolateNormals(false)
{
  createTriangleNormals();
  buildWaldBikkerData();
  buildTree();
  buildTriangleTable();
  mSurfaceArea = computeSurfaceArea();
  mOneOverSurfaceArea = 1.0f / mSurfaceArea;
} // end Mesh::Mesh()

Mesh
  ::Mesh(const std::vector<Point> &vertices,
         const std::vector<ParametricCoordinates> &parametrics,
         const std::vector<Triangle> &triangles)
    :Parent0(vertices,parametrics,triangles),Parent1(),mInterpolateNormals(false)
{
  createTriangleNormals();
  buildWaldBikkerData();
  buildTree();
  buildTriangleTable();
  mSurfaceArea = computeSurfaceArea();
  mOneOverSurfaceArea = 1.0f / mSurfaceArea;
} // end Mesh::Mesh()

Mesh
  ::Mesh(const std::vector<Point> &vertices,
         const std::vector<ParametricCoordinates> &parametrics,
         const std::vector<Normal> &normals,
         const std::vector<Triangle> &triangles)
    :Parent0(vertices,parametrics,normals,triangles),Parent1(),mInterpolateNormals(true)
{
  buildWaldBikkerData();
  buildTree();
  buildTriangleTable();
  mSurfaceArea = computeSurfaceArea();
  mOneOverSurfaceArea = 1.0f / mSurfaceArea;
} // end Mesh::Mesh()

void Mesh
  ::createTriangleNormals(void)
{
  mNormals.resize(mTriangles.size());

  for(size_t i = 0; i != mTriangles.size(); ++i)
  {
    Triangle tri = mTriangles[i];

    const Point &v0 = mPoints[tri[0]];
    const Point &v1 = mPoints[tri[1]];
    const Point &v2 = mPoints[tri[2]];

    mNormals[i] = (v1 - v0).cross(v2 - v0).normalize();
  } // end for i
} // end Mesh::createTriangleNormals()

Mesh
  ::~Mesh(void)
{
  ;
} // end Mesh::~Mesh()

bool Mesh
  ::intersect(const Ray &r) const
{
  // create a TriangleShadower
  TriangleShadower shadower = {this};
  return mTree.shadow(r.getAnchor(), r.getDirection(), r.getInterval()[0], r.getInterval()[1], shadower);
} // end Mesh::intersect()

bool Mesh
  ::intersect(const Ray &r, float &t, DifferentialGeometry &dg) const
{
  // create a TriangleIntersector
  TriangleIntersector intersector;
  intersector.init();
  intersector.mMesh = this;

  if(mTree.intersect(r.getAnchor(), r.getDirection(),
                     r.getInterval()[0], r.getInterval()[1],
                     intersector))
  {
    t = intersector.mT;
    const Triangle &tri = *intersector.mHitFace;

    // fill out DifferentialGeometry details
    getDifferentialGeometry(tri, r(t),
                            intersector.mBarycentricCoordinates[0],
                            intersector.mBarycentricCoordinates[1],
                            dg);
    return true;
  } // end if

  return false;
} // end Mesh::intersect()

float Mesh::TriangleBounder
  ::operator()(unsigned int axis, bool min, const Triangle *t)
{
  float result = std::numeric_limits<float>::infinity();
  if(!min) result = -result;

  // iterate through each adjacent vertex & pick the minimal/maximal
  const PointList &points = mMesh->getPoints();

  for(unsigned int i = 0; i < 3; ++i)
  {
    const Point &pos = points[(*t)[i]];

    if(min)
    {
      if(pos[axis] < result) result = pos[axis];
    } // end if
    else
    {
      if(pos[axis] > result) result = pos[axis];
    } // end else
  } // end for i

  return result;
} // end Mesh::TriangleBounder::operator()()

void Mesh
  ::getBoundingBox(BoundingBox &b) const
{
  b.setEmpty();

  // iterate over positions
  for(PointList::const_iterator p = getPoints().begin();
      p != getPoints().end();
      ++p)
  {
    b.addPoint(*p);
  } // end for p
} // end Mesh::getBoundingBox()

void Mesh
  ::buildTree(void)
{
  TriangleBounder bounder;
  bounder.mMesh = this;

  std::vector<const Triangle*> trianglePointers;
  for(TriangleList::const_iterator t = getTriangles().begin();
      t != getTriangles().end();
      ++t)
  {
    trianglePointers.push_back(&(*t));
  } // end for t

  mTree.buildTree(trianglePointers.begin(), trianglePointers.end(), bounder);
} // end Mesh::buildTree()

void Mesh
  ::getDifferentialGeometry(const Triangle &tri,
                            const Point &p,
                            const float b1,
                            const float b2,
                            DifferentialGeometry &dg) const
{
  // shorthand
  const Point &p1 = mPoints[tri[0]];
  const Point &p2 = mPoints[tri[1]];
  const Point &p3 = mPoints[tri[2]];

  // compute the last barycentric coordinate
  float b0 = 1.0f - b1 - b2;

  // XXX why do we do this here and not getDifferentialGeometry()?
  // interpolate normal?
  Normal ng;
  if(mInterpolateNormals)
  {
    // get each vertex's Normals
    const Mesh::NormalList &norms = getNormals();
    const Normal &n1 = norms[tri[0]];
    const Normal &n2 = norms[tri[1]];
    const Normal &n3 = norms[tri[2]];
    ng = b0 * n1 + b1 * n2 + b2 * n3;

    // no need to normalize this if they are
    // unit-length to begin with
    //ng = ng.normalize();
  } // end else
  else
  {
    // XXX just look up from mNormals
    ng = (p2 - p1).cross(p3 - p1).normalize();
  } // end else

  ParametricCoordinates uv0;
  ParametricCoordinates uv1;
  ParametricCoordinates uv2;
  getParametricCoordinates(tri, uv0, uv1, uv2);

  // compute deltas for partial derivatives
  float du1 = uv0[0] - uv2[0];
  float du2 = uv1[0] - uv2[0];
  float dv1 = uv0[1] - uv2[1];
  float dv2 = uv1[1] - uv2[1];
  Vector dp1 = p1 - p3, dp2 = p2 - p3;
  float determinant = du1 * dv2 - dv1 * du2;
  Vector dpdu, dpdv;
  if(determinant == 0.0)
  {
    // handle zero determinant case
    dpdu = ng.orthogonalVector();
    dpdv = ng.cross(dpdu).normalize();
  } // end if
  else
  {
    float invDet = 1.0f / determinant;
    dpdu = ( dv2*dp1 - dv1*dp2) * invDet;
    dpdv = (-du2*dp1 + du1*dp2) * invDet;
  } // end else

  dg.setPointPartials(dpdu,dpdv);

  // interpolate uv using barycentric coordinates
  ParametricCoordinates uv;
  uv[0] = b0*uv0[0] + b1*uv1[0] + b2*uv2[0];
  uv[1] = b0*uv0[1] + b1*uv1[1] + b2*uv2[1];

  dg.setPoint(p);
  dg.setNormal(ng);
  dg.setParametricCoordinates(uv);
  dg.setTangent(dpdu.normalize());

  // force an orthonormal basis
  dg.setBinormal(ng.cross(dg.getTangent()));

  // set the inverse surface area
  dg.setInverseSurfaceArea(getInverseSurfaceArea());
} // end Mesh::getDifferentialGeometry()

void Mesh::TriangleIntersector
  ::init(void)
{
  mT = std::numeric_limits<float>::infinity();
  mHitFace = 0;
  mMesh = 0;
} // end TriangleIntersector::init()

void Mesh
  ::buildTriangleTable(void)
{
  std::vector<const Triangle*> pointers;
  std::vector<float> area;
  for(TriangleList::const_iterator t = mTriangles.begin();
      t != mTriangles.end();
      ++t)
  {
    pointers.push_back(&(*t));
    area.push_back(Parent0::computeSurfaceArea(*t));
  } // end for t

  mTriangleTable.build(pointers.begin(), pointers.end(),
                       area.begin(), area.end());
} // end Mesh::buildTriangleTable()

float Mesh
  ::computeSurfaceArea(void) const
{
  float result = 0;

  for(TriangleList::const_iterator t = mTriangles.begin();
      t != mTriangles.end();
      ++t)
  {
    result += Parent0::computeSurfaceArea(*t);
  } // end for t

  return result;
} // end Mesh::computeSurfaceArea()

void Mesh
  ::sampleSurfaceArea(const float u1,
                      const float u2,
                      const float u3,
                      DifferentialGeometry &dg,
                      float &pdf) const
{
  const Triangle &t = *mTriangleTable(u1);

  // XXX it seems like we ought to do this
  //     otherwise, we don't take into account the
  //     pdf of choosing t
  //const Triangle &t = *mTriangleTable(u1, pdf);

  const Point &v0 = mPoints[t[0]];
  const Point &v1 = mPoints[t[1]];
  const Point &v2 = mPoints[t[2]];

  Point p;
  ParametricCoordinates b;

  // XXX why don't we get the pdf here??
  UnitSquareToTriangle::evaluate(u2,u3, v0, v1, v2, p, b);

  // we need to send barycentrics here, not u2 and u3
  // as they are NOT the same thing
  getDifferentialGeometry(t, p, b[0], b[1], dg);

  // evaluate the pdf
  pdf = evaluateSurfaceAreaPdf(dg);
} // end Mesh::sampleSurfaceArea()

float Mesh
  ::evaluateSurfaceAreaPdf(const DifferentialGeometry &dg) const
{
  // we assume dg is actually on the surface of this Mesh
  return mOneOverSurfaceArea;
} // end Mesh::evaluateSurfaceAreaPdf()

float Mesh
  ::getSurfaceArea(void) const
{
  return mSurfaceArea;
} // end Mesh::getSurfaceArea()

float Mesh
  ::getInverseSurfaceArea(void) const
{
  return mOneOverSurfaceArea;
} // end Mesh::getInverseSurfaceArea()

void Mesh
  ::getWaldBikkerData(const Triangle &tri,
                      WaldBikkerData &data) const
{
  unsigned int k;
  buildWaldBikkerIntersectionData<gpcpu::float3,float>(mPoints[tri[0]],
                                                       mPoints[tri[1]],
                                                       mPoints[tri[2]],
                                                       data.mN,
                                                       k,
                                                       data.mBn[0], data.mBn[1],
                                                       data.mCn[0], data.mCn[1]);

  data.mDominantAxis = k;
  data.mUAxis = (data.mDominantAxis + 1) % 3;
  data.mVAxis = (data.mDominantAxis + 2) % 3;
} // end Mesh::getWaldBikkerData()

void Mesh
  ::buildWaldBikkerData(void)
{
  mWaldBikkerTriangleData.clear();

  // see http://www.devmaster.net/articles/raytracing_series/part7.php for details
  for(size_t i = 0; i < mTriangles.size(); ++i)
  {
    const Triangle &tri = mTriangles[i];

    WaldBikkerData data;
    getWaldBikkerData(tri, data);

    mWaldBikkerTriangleData.push_back(data);
  } // end for i
} // end Mesh::buildWaldBikkerData()

