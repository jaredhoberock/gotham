/*! \file BoundingVolumeHierarchy.inl
 *  \author Jared Hoberock
 *  \brief Inline file for BoundingVolumeHierarchy.h.
 */

#include "BoundingVolumeHierarchy.h"
#include <bittricks/bittricks.h>
#include <iostream>
#include <limits>
#include <algorithm>

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  const typename BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>::NodeIndex BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>::NULL_NODE = std::numeric_limits<NodeIndex>::max();

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  typename BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>::NodeIndex
    BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>
      ::addNode(const NodeIndex parent)
{
  NodeIndex result = NodeIndex(mParentIndices.size());

  mParentIndices.resize(mParentIndices.size() + 1);
  mChildIndices.resize(mChildIndices.size() + 1);
  mMinBoundHitIndex.resize(mMinBoundHitIndex.size() + 1);
  mMaxBoundMissIndex.resize(mMaxBoundMissIndex.size() + 1);
  setParentIndex(result, parent);
  setHitIndex(result, NULL_NODE);
  setMissIndex(result, NULL_NODE);

  return result;
} // end BoundingVolumeHierarchy::addNode()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  void BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>
      ::setBounds(const NodeIndex node,
                  const Point &m,
                  const Point &M)
{
  mMinBoundHitIndex[node][0] = m[0];
  mMinBoundHitIndex[node][1] = m[1];
  mMinBoundHitIndex[node][2] = m[2];

  mMaxBoundMissIndex[node][0] = M[0];
  mMaxBoundMissIndex[node][1] = M[1];
  mMaxBoundMissIndex[node][2] = M[2];
} // end BoundingVolumeHierarchy::setBounds()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  void BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>
      ::setChildren(const NodeIndex node,
                    const NodeIndex left,
                    const NodeIndex right)
{
  mChildIndices[node] = gpcpu::size2(left,right);
} // end BoundingVolumeHierarchy::setChildren()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  void BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>
      ::setHitIndex(const NodeIndex node,
                    const NodeIndex hit)
{
  mMinBoundHitIndex[node][3] = uintAsFloat(hit);
} // end BoundingVolumeHierarchy::setHitIndex()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  typename BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>::NodeIndex
    BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>
      ::getHitIndex(const NodeIndex n) const
{
  return floatAsUint(mMinBoundHitIndex[n][3]);
} // end NodeIndex::getMissIndex()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  void BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>
      ::setMissIndex(const NodeIndex node,
                     const NodeIndex miss)
{
  mMaxBoundMissIndex[node][3] = uintAsFloat(miss);
} // end BoundingVolumeHierarchy::setMissIndex()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  bool BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>
    ::intersectBox(const Point &o, const Point &invDir,
                   const Point &minBounds, const Point &maxBounds,
                   const Real &tMin, const Real &tMax)
{
  Point tMin3, tMax3;
  tMin3 = (minBounds - o) * invDir;
  tMax3 = (maxBounds - o) * invDir;

  Point tNear3(std::min(tMin3[0], tMax3[0]),
               std::min(tMin3[1], tMax3[1]),
               std::min(tMin3[2], tMax3[2]));
  Point  tFar3(std::max(tMin3[0], tMax3[0]),
               std::max(tMin3[1], tMax3[1]),
               std::max(tMin3[2], tMax3[2]));

  Real tNear = std::max(std::max(tNear3[0], tNear3[1]), tNear3[2]);
  Real tFar  = std::min(std::min( tFar3[0],  tFar3[1]),  tFar3[2]);

  bool hit = tNear <= tFar;
  return hit && tMax >= tNear && tMin <= tFar;
} // end intersectBox()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  template<typename Intersector>
    bool BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>
      ::intersect(const Point &o, const Point &d,
                  Real tMin, Real tMax,
                  Intersector &intersect) const
{
  Point invDir;
  invDir[0] = Real(1.0) / d[0];
  invDir[1] = Real(1.0) / d[1];
  invDir[2] = Real(1.0) / d[2];

  NodeIndex currentNode = mRootIndex;
  NodeIndex hitIndex;
  NodeIndex missIndex;
  bool hit = false;
  bool result = false;
  float t = tMax;
  while(currentNode != NULL_NODE)
  {
    hitIndex = getHitIndex(currentNode);
    missIndex = getMissIndex(currentNode);

    // leaves (primitives) are listed before interior nodes
    // so bounding boxes occur after the root index
    if(currentNode >= mRootIndex)
    {
      hit = intersectBox(o, invDir,
                         getMinBounds(currentNode),
                         getMaxBounds(currentNode),
                         tMin, tMax);
    } // end if
    else
    {
      // XXX we can potentially ignore the checks on tMax and tMin
      //     here if we require the Intersector to do it
      hit = intersect(o,d,currentNode,t) && t < tMax && t > tMin;
      result |= hit;
      if(hit)
        tMax = std::min(t, tMax);
    } // end else

    currentNode = hit ? hitIndex : missIndex;
  } // end while

  return result;
} // end BoundingVolumeHierarchy::intersect()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  template<typename Bounder>
    void BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>
      ::findBounds(const std::vector<size_t>::iterator begin,
                   const std::vector<size_t>::iterator end,
                   const std::vector<Primitive> &primitives,
                   CachedBounder<Bounder> &bound,
                   Point &m, Point &M)
{
  Real inf = std::numeric_limits<Real>::infinity();
  Point inf3(inf,inf,inf);
  m = inf3;
  M = -inf3;

  Real x;
      
  for(std::vector<size_t>::iterator t = begin;
      t != end;
      ++t)
  {
    for(size_t i =0;
        i < 3;
        ++i)
    {
      x = bound(i, true, *t);

      if(x < m[i])
      {
        m[i] = x;
      } // end if

      x = bound(i, false, *t);

      if(x > M[i])
      {
        M[i] = x;
      } // end if
    } // end for j
  } // end for t

  // always widen the bounding box
  // this ensures that axis-aligned primitives always
  // lie strictly within the bounding box
  for(size_t i = 0; i != 3; ++i)
  {
    m[i] -= EPS;
    M[i] += EPS;
  } // end for i
} // end BoundingVolumeHierarchy::findBounds()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  template<typename Bounder>
    void BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>
      ::build(const std::vector<Primitive> &primitives,
              Bounder &bound)
{
  // clear nodes first
  clear();

  // we will sort an array of indices
  std::vector<size_t> primIndices(primitives.size());
  for(size_t i = 0; i != primitives.size(); ++i)
  {
    primIndices[i] = i;
  } // end for i

  // initialize
  // Leaf nodes come at the beginning of the list of nodes
  // Create as many as we have primitives
  for(size_t i = 0; i != primitives.size(); ++i)
  {
    // we don't know the leaf node's parent at first,
    // so set it to NULL_NODE for now
    addNode(NULL_NODE);
  } // end for i

  // create a CachedBounder
  CachedBounder<Bounder> cachedBound(bound,primitives);

  // recurse
  mRootIndex = build(NULL_NODE,
                     primIndices.begin(),
                     primIndices.end(),
                     primitives,
                     cachedBound);

  // for each node, compute the index of the
  // next node in a hit/miss ray traversal
  NodeIndex miss,hit;
  for(NodeIndex i = 0;
      i != NodeIndex(getNumNodes());
      ++i)
  {
    if(getParentIndex(i) == NULL_NODE && i != mRootIndex)
    {
      assert(0);
    } // end if

    hit = computeHitIndex(i);
    miss = computeMissIndex(i);
    setHitIndex(i, hit);
    setMissIndex(i, miss);
  } // end for i
} // end BoundingVolumeHierarchy::build()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  template<typename Bounder>
    typename BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>::NodeIndex
      BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>
        ::build(const NodeIndex parent,
                std::vector<size_t>::iterator begin,
                std::vector<size_t>::iterator end,
                const std::vector<PrimitiveType> &primitives,
                Bounder &bound)
{
  if(begin + 1 == end)
  {
    // we've hit a leaf
    NodeIndex result = *begin;

    // set its parent
    setParentIndex(result, parent);

    // set its children to null
    setChildren(result, NULL_NODE, NULL_NODE);
    return result;
  } // end if
  else if(begin == end)
  {
    std::cerr << "BoundingVolumeHierarchy::build(): empty base case." << std::endl;
    return NULL_NODE;
  } // end else if
  
  // find the bounds of the Primitives
  Point m, M;
  findBounds(begin, end, primitives, bound, m, M);

  // create a new node
  NodeIndex index = addNode(parent);
  setBounds(index, m, M);
  
  size_t axis = findPrincipalAxis(m, M);

  // create an ordering
  PrimitiveSorter<Bounder> sorter(axis,primitives,bound);
  
  // sort the median
  std::vector<size_t>::iterator split
    = begin + (end - begin) / 2;

  std::nth_element(begin, split, end, sorter);

  NodeIndex leftChild = build(index, begin, split,
                              primitives, bound);
  NodeIndex rightChild = build(index, split, end,
                               primitives, bound);

  setChildren(index, leftChild, rightChild);
  setMissIndex(index, NULL_NODE);

  return index;
} // end BoundingVolumeHierarchy::build()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  size_t BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>
    ::findPrincipalAxis(const Point &min,
                        const Point &max)
{
  // find the principal axis of the points
  size_t axis = 4;
  float maxLength = -std::numeric_limits<Real>::infinity();
  float temp;
  for(size_t i = 0; i < 3; ++i)
  {
    temp = max[i] - min[i];
    if(temp > maxLength)
    {
      maxLength = temp;
      axis = i;
    } // end if
  } // end for

  return axis;
} // end BoundingVolumeHierarchy::findPrincipalAxis()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  typename BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>::NodeIndex
    BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>
      ::computeHitIndex(const NodeIndex i) const
{
  // case 1
  // return the left index if we are an interior node
  NodeIndex result = getLeftIndex(i);
  
  if(result == NULL_NODE)
  {
    // case 1
    // the next node to visit after a hit is our right brother, 
    // if he exists
    result = computeRightBrotherIndex(i);
    if(result == NULL_NODE)
    {
      // if we have no right brother, return my parent's
      // miss index
      result = computeMissIndex(getParentIndex(i));
    } // end if
  } // end if

  return result;
} // end BoundingVolumeHierarchy::computeMissIndex()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  typename BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>::NodeIndex
    BoundingVolumeHierarchy<PrimitiveType, PointType, RealType>
      ::computeMissIndex(const NodeIndex i) const
{
  NodeIndex result = mRootIndex;
  
  // case 1
  // there is no next node to visit after the root
  if(i == mRootIndex)
  {
    result = NULL_NODE;
  } // end if
  else
  {
    // case 2
    // if i am my parent's left child, return my brother
    result = computeRightBrotherIndex(i);
    if(result == NULL_NODE)
    {
      // case 3
      // return my parent's miss index
      result = computeMissIndex(getParentIndex(i));
    } // end if
  } // end else

  return result;
} // end BoundingVolumeHierarchy::computeMissIndex()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  typename BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>::NodeIndex
    BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>
      ::computeRightBrotherIndex(const NodeIndex i) const
{
  NodeIndex result = NULL_NODE;
  NodeIndex parent = getParentIndex(i);

  if(i == getLeftIndex(parent))
  {
    result = getRightIndex(parent);
  } // end if

  return result;
} // BoundingVolumeHierarchy::computeRightBrotherIndex()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  typename BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>::NodeIndex
    BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>
      ::getRootIndex(void) const
{
  return mRootIndex;
} // end BoundingVolumeHierarchy::getRootIndex()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  typename BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>::NodeIndex
    BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>
      ::getParentIndex(const NodeIndex n) const
{
  return mParentIndices[n];
} // end NodeIndex::getParentIndex()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  typename BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>::NodeIndex
    BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>
      ::getLeftIndex(const NodeIndex n) const
{
  return mChildIndices[n][0];
} // end NodeIndex::getLeftIndex()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  typename BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>::NodeIndex
    BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>
      ::getRightIndex(const NodeIndex n) const
{
  return mChildIndices[n][1];
} // end NodeIndex::getRightIndex()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  typename BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>::NodeIndex
    BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>
      ::getMissIndex(const NodeIndex n) const
{
  return floatAsUint(mMaxBoundMissIndex[n][3]);
} // end NodeIndex::getMissIndex()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  void BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>
    ::setParentIndex(const NodeIndex n,
                     const NodeIndex parent)
{
  mParentIndices[n] = parent;
} // end NodeIndex::getParentIndex()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  const PointType &BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>
    ::getMinBounds(const NodeIndex n) const
{
  return *reinterpret_cast<const PointType*>(&mMinBoundHitIndex[n]);
} // end BoundingVolumeHierarchy::getMinBounds()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  const PointType &BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>
    ::getMaxBounds(const NodeIndex n) const
{
  return *reinterpret_cast<const PointType*>(&mMaxBoundMissIndex[n]);
} // end BoundingVolumeHierarchy::getMaxBounds()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  void BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>
    ::clear(void)
{
  mParentIndices.clear();
  mChildIndices.clear();
  mMinBoundHitIndex.clear();
  mMaxBoundMissIndex.clear();
} // end BoundingVolumeHierarchy::clear()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  size_t BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>
    ::getNumNodes(void) const
{
  return mMinBoundHitIndex.size();
} // end BoundingVolumeHierarchy::getNumNodes()

template<typename PrimitiveType,
         typename PointType,
         typename RealType>
  template<typename Bounder>
    BoundingVolumeHierarchy<PrimitiveType,PointType,RealType>::CachedBounder<Bounder>
      ::CachedBounder(Bounder &bound,
                      const std::vector<Primitive> &primitives)
{
  mPrimMinBounds[0].resize(primitives.size());
  mPrimMinBounds[1].resize(primitives.size());
  mPrimMinBounds[2].resize(primitives.size());

  mPrimMaxBounds[0].resize(primitives.size());
  mPrimMaxBounds[1].resize(primitives.size());
  mPrimMaxBounds[2].resize(primitives.size());

  size_t i = 0;
  for(typename std::vector<PrimitiveType>::const_iterator prim = primitives.begin();
      prim != primitives.end();
      ++prim, ++i)
  {
    mPrimMinBounds[0][i] = bound(0, true, *prim);
    mPrimMinBounds[1][i] = bound(1, true, *prim);
    mPrimMinBounds[2][i] = bound(2, true, *prim);

    mPrimMaxBounds[0][i] = bound(0, false, *prim);
    mPrimMaxBounds[1][i] = bound(1, false, *prim);
    mPrimMaxBounds[2][i] = bound(2, false, *prim);
  } // end for prim
} // end CachedBounder::CachedBounder()

