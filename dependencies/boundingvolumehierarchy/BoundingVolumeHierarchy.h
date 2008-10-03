/*! \file BoundingVolumeHierarchy.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class representing a
 *         hierarchy of axis aligned bounding boxes.
 */

#ifndef BOUNDING_VOLUME_HIERARCHY_H
#define BOUNDING_VOLUME_HIERARCHY_H

#include <vector>
#include <gpcpu/Vector.h>

template<typename PrimitiveType, typename PointType, typename RealType = float>
  class BoundingVolumeHierarchy
{
  public:
    typedef PrimitiveType Primitive;
    typedef PointType Point;
    typedef RealType Real;

    typedef unsigned int NodeIndex;
    static const NodeIndex NULL_NODE;
    static const float EPS;

    template<typename Bounder>
      void build(const std::vector<Primitive> &primitives,
                 Bounder &bound);

    template<typename Intersector>
      bool intersect(const Point &o, const Point &d,
                     Real tMin, Real tMax,
                     Intersector &intersect) const;

    static bool intersectBox(const Point &o, const Point &invDir,
                             const Point &minBounds, const Point &maxBounds, 
                             const Real &tMin, const Real &tMax);

    /*! This method returns mRootIndex.
     *  \return mRootIndex
     */
    inline NodeIndex getRootIndex(void) const;

    /*! This method returns the number of Nodes in this hierarchy.
     *  \return The number of Nodes.
     */
    inline size_t getNumNodes(void) const;

  protected:
    /*! This method clears this BoundingVolumeHierarchy of Nodes.
     */
    inline void clear(void);

    /*! This method adds a new Node to this hierarchy.
     *  \param parent The parent of the Node to add.
     *  \return The index of the node.
     */
    NodeIndex addNode(const NodeIndex parent);

    /*! This method sets the bounds of the given Node.
     *  \param n The NodeIndex of the Node of interest.
     *  \param m The minimal corner of the Node.
     *  \param M The maximal corner of the Node.
     */
    void setBounds(const NodeIndex node,
                   const Point &m,
                   const Point &M);

    /*! This method sets the child pointers of the given Node.
     *  \param n The NodeIndex of the Node of interest.
     *  \param left The NodeIndex of the left child.
     *  \param right The NodeIndex of the right child.
     */
    void setChildren(const NodeIndex node,
                     const NodeIndex left,
                     const NodeIndex right);

    /*! This method sets the hit pointer of the given Node.
     *  \param node The NodeIndex of the Node of interest.
     *  \param hit  The NodeIndex of the hit node.
     */
    void setHitIndex(const NodeIndex node,
                     const NodeIndex hit);

    /*! This method sets the miss pointer of the given Node.
     *  \param node The NodeIndex of the Node of interest.
     *  \param miss The NodeIndex of the miss Node.
     */
    void setMissIndex(const NodeIndex node,
                      const NodeIndex miss);

    /*! This method returns the NodeIndex of the given Node's parent.
     *  \param n The NodeIndex of the Node of interest.
     *  \return The NodeIndex of n's parent.
     */
    inline NodeIndex getParentIndex(const NodeIndex node) const;

    /*! This method returns the NodeIndex of the given Node's left child.
     *  \param n The NodeIndex of the Node of interest.
     *  \return The NodeIndex of n's left child.
     */
    inline NodeIndex getLeftIndex(const NodeIndex node) const;

    /*! This method returns the NodeIndex of the given Node's right child.
     *  \param n The NodeIndex of the Node of interest.
     *  \return The NodeIndex of n's right child.
     */
    inline NodeIndex getRightIndex(const NodeIndex node) const;

    /*! This method returns the NodeIndex of the given Node's miss Node.
     *  \param n The NodeIndex of the Node of interest.
     *  \return The NodeIndex of n's hit Node.
     */
    inline NodeIndex getHitIndex(const NodeIndex node) const;

    /*! This method returns the NodeIndex of the given Node's miss Node.
     *  \param n The NodeIndex of the Node of interest.
     *  \return The NodeIndex of n's miss Node.
     */
    inline NodeIndex getMissIndex(const NodeIndex node) const;

    /*! This method returns the minimal bounds of the given Node's bounding box.
     *  \param n The NodeIndex of the Node of interest.
     *  \return The min corner of n's bounding box.
     */
    inline const Point &getMinBounds(const NodeIndex n) const;

    /*! This method returns the maximal bounds of the given Node's bounding box.
     *  \param n The NodeIndex of the Node of interest.
     *  \return The max corner of n's bounding box.
     */
    inline const Point &getMaxBounds(const NodeIndex n) const;

    /*! This method sets the NodeIndex of the given Node's parent.
     *  \param n The NodeIndex of the Node of interest.
     *  \param parent The NodeIndex of n's parent.
     */
    inline void setParentIndex(const NodeIndex node,
                               const NodeIndex parent);

    /*! The idea of this class is to wrap Bounder
     *  and accelerate build() by caching the results
     *  of Bounder.
     *
     *  This gives about a 10x build speedup on the bunny
     *  in Cornell box scene.
     */
    template<typename Bounder>
      class CachedBounder
    {
      public:
        inline CachedBounder(Bounder &bound,
                             const std::vector<Primitive> &primitives);

        inline float operator()(const size_t axis,
                                const bool min,
                                const size_t primIndex)
        {
          if(min)
          {
            return mPrimMinBounds[axis][primIndex];
          }

          return mPrimMaxBounds[axis][primIndex];
        } // end operator()()

      protected:
        std::vector<RealType> mPrimMinBounds[3];
        std::vector<RealType> mPrimMaxBounds[3];
    }; // end CachedBounder

    template<typename Bounder>
      static void findBounds(const std::vector<size_t>::iterator begin,
                             const std::vector<size_t>::iterator end,
                             const std::vector<Primitive> &primitives,
                             CachedBounder<Bounder> &bound,
                             Point &m, Point &M);

    template<typename Bounder>
      struct PrimitiveSorter
    {
      inline PrimitiveSorter(const size_t axis,
                             const std::vector<PrimitiveType> &primitives,
                             Bounder &bound)
        :mAxis(axis),mPrimitives(primitives),mBound(bound)
      {
        ;
      } // end PrimitiveSorter()

      inline bool operator()(const size_t lhs, const size_t rhs) const
      {
        return mBound(mAxis, true, lhs) < mBound(mAxis, true, rhs);
      } // end operator<()

      size_t mAxis;
      const std::vector<PrimitiveType> &mPrimitives;
      Bounder &mBound;
    }; // end PrimitiveSorter

    template<typename Bounder>
      NodeIndex build(const NodeIndex parent,
                      std::vector<size_t>::iterator begin,
                      std::vector<size_t>::iterator end,
                      const std::vector<PrimitiveType> &primitives,
                      Bounder &bound);

    static size_t findPrincipalAxis(const Point &min,
                                          const Point &max);

    /*! This method computes the index of the next node in a
     *  depth first traversal of this tree, from node i.
     *  \param i The Node of interest.
     *  \return The index of the next Node from i, if it exists;
     *          NULL_NODE, otherwise.
     */
    NodeIndex computeHitIndex(const NodeIndex i) const;

    /*! This method computes the index of the next node in a
     *  depth first traversal of this tree, from node i.
     *  \param i The Node of interest.
     *  \return The index of the next Node from i, if it exists;
     *          NULL_NODE, otherwise.
     */
    NodeIndex computeMissIndex(const NodeIndex i) const;

    /*! This method computes the index of a Node's brother to the right,
     *  if it exists.
     *  \param i The index of the Node of interest.
     *  \return The index of Node i's brother to the right, if it exists;
     *          NULL_NODE, otherwise.
     */
    NodeIndex computeRightBrotherIndex(const NodeIndex i) const;

    std::vector<NodeIndex> mParentIndices;
    std::vector<gpcpu::size2> mChildIndices;

    // XXX these should probably vectors of the type of RealType
    std::vector<gpcpu::float4> mMinBoundHitIndex;
    std::vector<gpcpu::float4> mMaxBoundMissIndex;

    NodeIndex mRootIndex;
}; // end BoundingVolumeHierarchy

#include "BoundingVolumeHierarchy.inl"

#endif // BOUNDING_VOLUME_HIERARCHY_H

