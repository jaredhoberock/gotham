/*! \file Mesh.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Mesh.h.
 */

#include "Mesh.h"

bool Mesh::TriangleIntersector
  ::operator()(const Point &anchor, const Point &dir,
               const Triangle **begin, const Triangle **end,
               float minT, float maxT)
{
  float t;
  float b1,b2;
  while(begin != end)
  {
    //// intersect ray with object
    //if(mMesh->intersect(anchor, dir, **begin, *mMesh, t, b1, b2))
    //{
    //  if(t < mT && t >= minT && t <= maxT)
    //  {
    //    mT = t;
    //    mBarycentricCoordinates[0] = b1;
    //    mBarycentricCoordinates[1] = b2;
    //    mHitFace = &(**begin);
    //  } // end if
    //} // end if

    if(mMesh->intersectWaldBikker(anchor, dir, **begin, *mMesh, minT, maxT, t, b1, b2))
    {
      if(t < mT)
      {
        mT = t;
        mBarycentricCoordinates[0] = b1;
        mBarycentricCoordinates[1] = b2;
        mHitFace = &(**begin);
      } // end if
    } // end if

    //if(mMesh->intersectWaldBikker(anchor, dir, **begin, *mMesh, minT, mT, t, b1, b2))
    //{
    //  mT = t;
    //  mBarycentricCoordinates[0] = b1;
    //  mBarycentricCoordinates[1] = b2;
    //  mHitFace = &(**begin);
    //} // end if

    ++begin; 
  }// end while

  return mHitFace != 0;
} // end TriangleIntersector::operator()()

bool Mesh::TriangleShadower
  ::operator()(const Point &anchor, const Point &dir,
               const Triangle **begin, const Triangle **end,
               float minT, float maxT)
{
  float t = 0;
  float b1,b2;
  while(begin != end)
  {
    // intersect ray with object
    // XXX PERF: it may be more performant to let the intersection
    //           method allocate its own b1 & b2 rather than pass pointers
    if(mMesh->intersect(anchor, dir, **begin, *mMesh, t, b1, b2))
    {
      if(t >= minT && t <= maxT)
      {
        return true;
      } // end if
    } // end if

    //if(mMesh->intersectWaldBikker(anchor, dir, **begin, *mMesh, minT, maxT, t, b1, b2))
    //{
    //  return true;
    //} // end if

    ++begin; 
  }// end while

  return false;
} // end TriangleIntersector::operator()()

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

bool Mesh
  ::intersectWaldBikker(const Point &o,
                        const Vector &dir,
                        const Triangle &f,
                        const Mesh &m,
                        const float minT,
                        const float maxT,
                        float &t,
                        float &b1,
                        float &b2)
{
  // fetch the intersection data
  size_t i = &f - &*m.mTriangles.begin();
  const WaldBikkerData &data = m.mWaldBikkerTriangleData[i];

  // calculate distance to triangle plane
  float denom = (dir[data.mDominantAxis] + data.mN[1] * dir[data.mUAxis] + data.mN[2] * dir[data.mVAxis]);
  float numer = (data.mN[0] - o[data.mDominantAxis] - data.mN[1] * o[data.mUAxis] - data.mN[2] * o[data.mVAxis]);
  t = numer / denom;

  if(t < minT || t > maxT) return false;

  const Point &p = m.mPoints[f[0]];

  // calculate hit point
  float pu = o[data.mUAxis] + t * dir[data.mUAxis] - p[data.mUAxis];
  float pv = o[data.mVAxis] + t * dir[data.mVAxis] - p[data.mVAxis];
  b1 = pv * data.mBn[0] + pu * data.mBn[1];
  if(b1 < 0) return false;
  b2 = pu * data.mCn[0] + pv * data.mCn[1];
  if(b2 < 0) return false;

  if(b1 + b2 > 1.0f) return false;

  return true;
} // end Mesh::intersectWaldBikker()

void Mesh
  ::getParametricCoordinates(const Triangle &tri,
                             ParametricCoordinates &uv0,
                             ParametricCoordinates &uv1,
                             ParametricCoordinates &uv2) const
{
  if(mParametrics.size() != 0)
  {
    uv0 = mParametrics[tri[0]];
    uv1 = mParametrics[tri[1]];
    uv2 = mParametrics[tri[2]];
  } // end if
  else
  {
    uv0 = ParametricCoordinates(0,0);
    uv1 = ParametricCoordinates(0,1);
    uv2 = ParametricCoordinates(1,1);
  } // end else
} // end Mesh::getParametricCoordinates()
