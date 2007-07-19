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

// XXX implement this
bool Mesh
  ::intersect(const Ray &r) const
{
  // create a TriangleIntersector
  TriangleIntersector intersector;
  intersector.init();
  intersector.mMesh = this;

  return mTree.intersect(r.getAnchor(), r.getDirection(), r.getInterval()[0], r.getInterval()[1], intersector);
} // end Mesh::intersect()

// XXX implement this
bool Mesh
  ::intersect(const Ray &r, float &t, DifferentialGeometry &dg) const
{
  // create a TriangleIntersector
  TriangleIntersector intersector;
  intersector.init();
  intersector.mMesh = this;

  if(mTree.intersect(r.getAnchor(), r.getDirection(), r.getInterval()[0], r.getInterval()[1], intersector))
  {
    t = intersector.mT;
    const Triangle &tri = *intersector.mHitFace;

    // fill out DifferentialGeometry details
    // XXX fill out the rest of them
    dg.getPoint() = r(t);
    Vector3 e1 = mPoints[tri[1]] - mPoints[tri[0]];
    Vector3 e2 = mPoints[tri[2]] - mPoints[tri[0]];
    dg.getNormal() = e1.cross(e2).normalize();
    dg.setParametricCoordinates(intersector.mBarycentricCoordinates);
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

bool Mesh::TriangleIntersector
  ::operator()(const Point &anchor, const Point &dir,
               const Triangle **begin, const Triangle **end,
               float minT, float maxT)
{
  float t = 0;
  float b1,b2;
  while(begin != end)
  {
    // intersect ray with object
    if(mMesh->intersect(anchor, dir, **begin, *mMesh, t, b1, b2))
    {
      if(t < mT && t >= minT && t <= maxT)
      {
        mT = t;
        mBarycentricCoordinates[0] = b1;
        mBarycentricCoordinates[1] = b2;
        mHitFace = &(**begin);
      } // end if
    } // end if

    ++begin; 
  }// end while

  return mHitFace != 0;
} // end TriangleIntersector::operator()()

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

bool Mesh
  ::intersect(const Point &o,
              const Vector &dir,
              const Triangle &f,
              const Mesh &m,
              float &t,
              DifferentialGeometry &dg)
{
  // shorthand
  const Point &p1 = m.mPoints[f[0]];
  const Point &p2 = m.mPoints[f[1]];
  const Point &p3 = m.mPoints[f[2]];

  Vector e1 = p2 - p1;
  Vector e2 = p3 - p1;
  Vector s1 = dir.cross(e2);
  float divisor = s1.dot(e1);
  if(divisor == 0.0f) return false;

  float invDivisor = 1.0f / divisor;

  // compute barycentric coordinates
  Vector d = o - p1;
  float b1 = d.dot(s1) * invDivisor;
  if(b1 < 0.0 || b1 > 1.0) return false;

  Vector s2 = d.cross(e1);
  float b2 = dir.dot(s2) * invDivisor;
  if(b2 < 0.0 || b1 + b2 > 1.0) return false;

  // compute t
  t = e2.dot(s2) * invDivisor;

  // compute the last barycentric coordinate
  float b0 = 1.0f - b1 - b2;

  // interpolate normal?
  Normal n(0,0,0);
  //if(f[0].mNormalIndex == -1 ||
  //   f[1].mNormalIndex == -1 ||
  //   f[2].mNormalIndex == -1)
  //{
  n = e1.cross(e2);
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
  n = n.normalize();

  // XXX implement this!
  Vector3 dpdu(0,0,0), dpdv(0,0,0);
  ParametricCoordinates uv;
  ParametricCoordinates uvs[3] = {ParametricCoordinates(0,0), ParametricCoordinates(0,1), ParametricCoordinates(1,1)};
  // compute deltas for partial derivatives
  float du1 = uvs[0][0] - uvs[2][0];
  float du2 = uvs[1][0] - uvs[2][0];
  float dv1 = uvs[0][1] - uvs[2][1];
  float dv2 = uvs[1][1] - uvs[2][1];
  Vector dp1 = p1 - p3, dp2 = p2 - p3;
  float determinant = du1 * dv2 - dv1 * du2;
  //if(determinant == 0.0)
  //{
  //  // handle zero determinant case
  //  ///\todo Fix later
  //} // end if
  //else
  //{
    float invDet = 1.0f / determinant;
    dpdu = ( dv2*dp1 - dv1*dp2) * invDet;
    dpdv = (-du2*dp1 + du1*dp2) * invDet;
  //} // end else

  // interpolate parametric coordinates using barycentric coordinates
  uv[0] = b0*uvs[0][0] + b1*uvs[1][0] + b2*uvs[2][0];
  uv[1] = b0*uvs[0][1] + b1*uvs[1][1] + b2*uvs[2][1];

  // return differential geometry in world coordinates
  Point p = o + t*dir;
  dg = DifferentialGeometry(p,
                            n,
                            dpdu, dpdv,
                            Vector(0,0,0), Vector(0,0,0), uv, &m);

  return true;
} // end Mesh::intersect()

bool Mesh
  ::intersect(const Point &o,
              const Vector &dir,
              const Triangle &f,
              const Mesh &m,
              float &t,
              float &b1,
              float &b2)
{
  // shorthand
  const Point &p1 = m.mPoints[f[0]];
  const Point &p2 = m.mPoints[f[1]];
  const Point &p3 = m.mPoints[f[2]];

  Vector e1 = p2 - p1;
  Vector e2 = p3 - p1;
  Vector s1 = dir.cross(e2);
  float divisor = s1.dot(e1);
  if(divisor == 0.0f)
  {
    return false;
  } // end if

  float invDivisor = 1.0f / divisor;

  // compute barycentric coordinates
  Vector d = o - p1;
  b1 = d.dot(s1) * invDivisor;
  if(b1 < 0.0 || b1 > 1.0)
  {
    return false;
  } // end if

  Vector s2 = d.cross(e1);
  b2 = dir.dot(s2) * invDivisor;
  if(b2 < 0.0 || b1 + b2 > 1.0)
  {
    return false;
  } // end if

  // compute t
  t = invDivisor * e2.dot(s2);

  return true;
} // end Mesh::intersect()

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

  UnitSquareToTriangle::evaluate(u2,u3, v0, v1, v2, dg.getPoint());

  // XXX implement shading normals
  // XXX implement derivatives
  dg.getNormal() = (v1 - v0).cross(v2 - v0).normalize();

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

