/*! \file PhotonMap.inl
 *  \author Jared Hoberock
 *  \brief Inline file for PhotonMap.h.
 */

#include "PhotonMap.h"
#include <limits>
#include <iostream>
#include <boost/progress.hpp>

PhotonMap
  ::PhotonMap(void)
{
  ;
} // end PhotonMap::PhotonMap()

PhotonMap
  ::PhotonMap(const std::vector<Point> &positions,
              const std::vector<Vector> &wi,
              const std::vector<Spectrum> &power)
{
  clear();

  size_t num = std::min(positions.size(), std::min(wi.size(), power.size()));
  for(size_t i = 0; i != num; ++i)
  {
    resize(size() + 1);
    back().mPoint = positions[i];
    back().mWi    = wi[i];
    back().mPower = power[i];
  } // end for i
} // end PhotonMap::PhotonMap()

void PhotonMap
  ::sort(void)
{
  mNodes.resize(size());
  sort(begin(), end());
  //sort(0, size());
} // end PhotonMap::sort()

bool PhotonMap::PhotonSorter
  ::operator()(const Photon &lhs, const Photon &rhs) const
{
  return lhs.mPoint[mAxis] < rhs.mPoint[mAxis];
} // end PhotonSorter::operator()()

void PhotonMap
  ::sort(const size_t b, const size_t e)
{
  if(b >= e) return;

  // find the bounds of the range
  Point m, M;
  findBounds(begin() + b, begin() + e, m, M);

  // choose the largest axis to split
  Vector diag = M - m;
  diag *= diag;

  unsigned int axis = 4;
  float val = -1.0;
  for(unsigned int i = 0; i < 3; ++i)
  {
    if(diag[i] > val)
    {
      val = diag[i];
      axis = i;
    } // end if
  } // end for i

  // sort on that axis
  PhotonSorter sorter = {axis};
  size_t mid = (e + b) / 2;
  std::nth_element(&(*this)[b], &(*this)[mid], &(*this)[e], sorter);

  // set the kd-node data
  mNodes[mid].mSplitAxis = axis;
  mNodes[mid].mSplitValue = (*this)[mid].mPoint[axis];

  // recurse
  sort(b, mid);
  sort(mid + 1,e);
} // end PhotonMap::sort()

void PhotonMap
  ::sort(const iterator &b, const iterator &e)
{
  int diff = static_cast<unsigned int>(e-b);
  if(diff <= 0) return;

  // find the bounds of the range
  Point m, M;
  findBounds(b, e, m, M);

  // choose the largest axis to split
  Vector diag = M - m;
  diag *= diag;

  unsigned int axis = 4;
  float val = -1.0;
  for(unsigned int i = 0; i < 3; ++i)
  {
    if(diag[i] > val)
    {
      val = diag[i];
      axis = i;
    } // end if
  } // end for i

  // sort on that axis
  PhotonSorter sorter = {axis};
  iterator mid = b + diff/2;
  std::nth_element(b, mid, e, sorter);

  // set the kd-node data
  mNodes[static_cast<unsigned int>(mid - begin())].mSplitAxis = axis;
  mNodes[static_cast<unsigned int>(mid - begin())].mSplitValue = mid->mPoint[axis];

  // recurse
  sort(b, mid);
  sort(mid + 1,e);
} // end PhotonMap::sort()

void PhotonMap
  ::findBounds(const iterator &b, const iterator &e,
               Point &m, Point &M)
{
  float inf = std::numeric_limits<float>::infinity();
  m = Point(inf,inf,inf);
  M = -m;

  for(iterator i = b;
      i != e;
      ++i)
  {
    for(unsigned int j = 0;
        j != 3;
        ++j)
    {
      if(i->mPoint[j] > m[j])
      {
        m[j] = i->mPoint[j];
      } // end if
      if(i->mPoint[j] < M[j])
      {
        M[j] = i->mPoint[j];
      } // end if
    } // end for j
  } // end for i
} // end PhotonMap::findBounds()

template<typename Visitor>
  void PhotonMap
    ::rangeQuery(const Point &p,
                 float &maxDist2,
                 Visitor &visitor) const
{
  rangeQuery(0, static_cast<unsigned int>(size()),
             p, maxDist2,
             visitor);
} // end PhotonMap::rangeQuery()

template<typename Visitor>
  void PhotonMap
    ::rangeQuery(const unsigned int b,
                 const unsigned int e,
                 const Point &p,
                 float &maxDist2,
                 Visitor &visitor) const
{
  int diff = static_cast<int>(e-b);
  if(diff <= 1) return;

  unsigned int node = b + diff/2;

  unsigned int axis = mNodes[node].mSplitAxis;
  float splitValue = mNodes[node].mSplitValue;

  float d2 = p[axis] - splitValue;
  d2 *= d2;

  if(p[axis] <= splitValue)
  {
    // go down the left side
    rangeQuery(b, node, p, maxDist2, visitor);
    if(d2 < maxDist2)
    {
      // the right side is close enough go down it
      rangeQuery(node + 1, e, p, maxDist2, visitor);
    } // end if
  } // end if
  else
  {
    // go down the right side
    rangeQuery(node + 1, e, p, maxDist2, visitor);
    if(d2 < maxDist2)
    {
      // the left side is close enough go down it
      rangeQuery(b, node, p, maxDist2, visitor);
    } // end if
  } // end else

  // visit the photon
  d2 = (p - operator[](node).mPoint).norm2();
  if(d2 <= maxDist2)
  {
    visitor(operator[](node), d2, maxDist2);
  } // end if
} // end PhotonMap::rangeQuery()

std::ostream &operator<<(std::ostream &os, const PhotonMap &pm)
{
  boost::progress_display progress(3 * pm.size());

  os << "Photons(";

  os << "(";
  for(PhotonMap::const_iterator p = pm.begin();
      p != pm.end();
      ++p)
  {
    os << p->mPoint[0] << ",";
    os << p->mPoint[1] << ",";

    if(p + 1 == pm.end())
    {
      os << p->mPoint[2];
    } // end if
    else
    {
      os << p->mPoint[2] << ",";
    } // end else
    ++progress;
  } // end for p
  os << "),";

  os << "(";
  for(PhotonMap::const_iterator p = pm.begin();
      p != pm.end();
      ++p)
  {
    os << p->mWi[0] << ",";
    os << p->mWi[1] << ",";

    if(p + 1 == pm.end())
    {
      os << p->mWi[2];
    } // end if
    else
    {
      os << p->mWi[2] << ",";
    } // end else

    ++progress;
  } // end for p
  os << "),";

  os << "(";
  for(PhotonMap::const_iterator p = pm.begin();
      p != pm.end();
      ++p)
  {
    os << p->mPower[0] << ",";
    os << p->mPower[1] << ",";

    if(p + 1 == pm.end())
    {
      os << p->mPower[2];
    } // end if
    else
    {
      os << p->mPower[2] << ",";
    } // end else

    ++progress;
  } // end for p
  os << ")";

  os << ")";

  // flush the stream
  os.flush();
  return os;
} // end operator<<()

