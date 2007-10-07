/*! \file Mesh.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Mesh class.
 */

#include "Mesh.h"
#include "../geometry/Normal.h"
#include <2dmapping/UnitSquareToTriangle.h>

Mesh
  ::Mesh(const std::vector<Point> &vertices,
         const std::vector<Triangle> &triangles)
     :Parent0(vertices,triangles),Parent1()
{
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
    :Parent0(vertices,parametrics,triangles),Parent1()
{
  buildWaldBikkerData();
  buildTree();
  buildTriangleTable();
  mSurfaceArea = computeSurfaceArea();
  mOneOverSurfaceArea = 1.0f / mSurfaceArea;
} // end Mesh::Mesh()

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
    Vector3 e1 = mPoints[tri[1]] - mPoints[tri[0]];
    Vector3 e2 = mPoints[tri[2]] - mPoints[tri[0]];
    getDifferentialGeometry(tri, r(t), e1.cross(e2).normalize(),
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
                            const Normal &ng,
                            const float b1,
                            const float b2,
                            DifferentialGeometry &dg) const
{
  // shorthand
  const Point &p1 = mPoints[tri[0]];
  const Point &p2 = mPoints[tri[1]];
  const Point &p3 = mPoints[tri[2]];

  Vector e1 = p2 - p1;
  Vector e2 = p3 - p1;

  // compute the last barycentric coordinate
  float b0 = 1.0f - b1 - b2;

  // interpolate normal?
  //if(f[0].mNormalIndex == -1 ||
  //   f[1].mNormalIndex == -1 ||
  //   f[2].mNormalIndex == -1)
  //{
  //} // end if
  //else
  //{
  //  // get each vertex's Normals
  //  const Mesh::NormalList &norms = m.getNormals();
  //  const Normal &n1 = norms[f[0].mNormalIndex];
  //  const Normal &n2 = norms[f[1].mNormalIndex];
  //  const Normal &n3 = norms[f[2].mNormalIndex];
  //  n = b0 * n1 + b1 * n2 + b2 * n3;
  //} // end else

  // normalize it
  //dg.getNormal() = dg.getNormal().normalize();

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
  if(determinant == 0.0)
  {
    // handle zero determinant case
    dg.getPointPartials()[0] = ng.orthogonalVector();
    dg.getPointPartials()[1] = ng.cross(dg.getPointPartials()[0]).normalize();
  } // end if
  else
  {
    float invDet = 1.0f / determinant;
    dg.getPointPartials()[0] = ( dv2*dp1 - dv1*dp2) * invDet;
    dg.getPointPartials()[1] = (-du2*dp1 + du1*dp2) * invDet;
  } // end else

  // interpolate uv using barycentric coordinates
  ParametricCoordinates uv;
  uv[0] = b0*uv0[0] + b1*uv1[0] + b2*uv2[0];
  uv[1] = b0*uv0[1] + b1*uv1[1] + b2*uv2[1];

  dg.setPoint(p);
  dg.setNormal(ng);
  dg.setParametricCoordinates(uv);
  dg.setTangent(dg.getPointPartials()[0].normalize());

  // force an orthonormal basis
  dg.setBinormal(ng.cross(dg.getTangent()));
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
  const Point &v0 = mPoints[t[0]];
  const Point &v1 = mPoints[t[1]];
  const Point &v2 = mPoints[t[2]];

  Point p;
  UnitSquareToTriangle::evaluate(u2,u3, v0, v1, v2, p);

  // XXX implement shading normals
  // XXX implement derivatives
  Normal ng = (v1 - v0).cross(v2 - v0).normalize();

  getDifferentialGeometry(t, p, ng, u2, u3, dg);

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
  ::buildWaldBikkerData(void)
{
  mWaldBikkerTriangleData.clear();

  // see http://www.devmaster.net/articles/raytracing_series/part7.php for details
  for(size_t i = 0; i < mTriangles.size(); ++i)
  {
    const Triangle &tri = mTriangles[i];

    // compute the triangle's normal
    Vector b = mPoints[tri[2]] - mPoints[tri[0]];
    Vector c = mPoints[tri[1]] - mPoints[tri[0]];
    Normal n = c.cross(b).normalize();

    WaldBikkerData data;

    // determine dominant axis
    if(fabsf(n[0]) > fabsf(n[1]))
    {
      if(fabsf(n[0]) > fabsf(n[2])) data.mDominantAxis = 0;
      else data.mDominantAxis = 2;
    } // end if
    else
    {
      if(fabsf(n[1]) > fabsf(n[2])) data.mDominantAxis = 1;
      else data.mDominantAxis = 2;
    } // end else

    int u = (data.mDominantAxis + 1) % 3;
    int v = (data.mDominantAxis + 2) % 3;

    data.mUAxis = u;
    data.mVAxis = v;

    data.mN[0] = n.dot(mPoints[tri[0]]) / n[data.mDominantAxis];
    data.mN[1] = n[u] / n[data.mDominantAxis];
    data.mN[2] = n[v] / n[data.mDominantAxis];

    float bnu =  b[u] / (b[u] * c[v] - b[v] * c[u]);
    float bnv = -b[v] / (b[u] * c[v] - b[v] * c[u]);
    data.mBn = gpcpu::float2(bnu, bnv);

    float cnu =  c[v] / (b[u] * c[v] - b[v] * c[u]);
    float cnv = -c[u] / (b[u] * c[v] - b[v] * c[u]);
    data.mCn = gpcpu::float2(cnu, cnv);

    mWaldBikkerTriangleData.push_back(data);
  } // end for i
} // end Mesh::buildWaldBikkerData()

