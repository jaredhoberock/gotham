/*! \file AxisAlignedHypercube.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a template class for n-dimensional axis aligned cubes.
 */

#ifndef AXIS_ALIGNED_HYPERCUBE_H
#define AXIS_ALIGNED_HYPERCUBE_H

/*! \class AxisAlignedHypercube
 *  \brief An AxisAlignedHypercube is a generalized n-dimensional hypercube.
 *         The template is parameterized on n and point type.
 *  \note This is actually a Hyperbox, not a Hypercube.
 */
template<typename PointType, typename RealType, unsigned int mNumDimensions> class AxisAlignedHypercube
{
  public:
    /*! Null constructor puts the min corner at +infinity and the max corner at -infinity.
     */
    inline AxisAlignedHypercube(void);

    /*! Constructor accepts a min corner and a max corner.
     *  \param min Sets mMinCorner.
     *  \param max Sets mMaxCorner.
     */
    inline AxisAlignedHypercube(const PointType &min, const PointType &max);

    /*! This method sets the members of this AxisAlignedHypercube by being provided bounds.
     *  \param min Sets mMinCorner.
     *  \param max Sets mMaxCorner.
     */
    inline void set(const PointType &min, const PointType &max);

    /*! This method returns mMinCorner.
     *  \return mMinCorner.
     */
    inline const PointType &getMinCorner(void) const;

    /*! This method returns mMaxCorner.
     *  \return mMaxCorner.
     */
    inline const PointType &getMaxCorner(void) const;

    /*! This method returns mMinCorner.
     *  \return mMinCorner.
     */
    inline PointType &getMinCorner(void);

    /*! This method returns mMaxCorner.
     *  \return mMaxCorner.
     */
    inline PointType &getMaxCorner(void);

    /*! This method "adds a Point" to this AxisAlignedHypercube.
     *  If the Point lies outside this AxisAlignedHypercube, we very conservatively expand the AxisAlignedHypercube to include the Point.
     *  \param p The Point to add.
     *  \return true if this AxisAlignedHypercube had to be expanded to include p; false, otherwise.
     */
    inline bool addPoint(const PointType &p);

    /*! This method sets this AxisAlignedHypercube to empty.
     */
    inline void setEmpty(void);

    /*! operator[]() method returns a reference to a corner.
     *  \param i Which corner to return: 0 for mMinCorner; mMaxCorner, otherwise.
     *  \return The indicated corner of this AxisAlignedBoundingBox.
     */
    inline PointType &operator[](const unsigned int i);

    /*! operator[]() method returns a const reference to a corner.
     *  \param i Which corner to return: 0 for mMinCorner; mMaxCorner, otherwise.
     *  \return The indicated corner of this AxisAlignedHypercube.
     */
    inline const PointType &operator[](const unsigned int i) const;

    /*! This method computes the length of this AxisAlignedHypercube's diagonal.
     *  \return The length of this AxisAlignedHypercube's diagonal.
     */
    inline RealType computeDiagonalLength(void) const;

    /*! This method computes the volume of this AxisAlignedHypercube.
     *  \return The volume of this AxisAlignedHypercube.
     */
    RealType computeVolume(void) const;

    /*! This method computes the centroid of this AxisAlignedHypercube.
     *  \return (getMinCorner() + getMaxCorner() / 2.0
     */
    inline PointType computeCentroid(void) const;

  protected:
    /*! This method sets mMinCorner.
     *  \param m Sets mMinCorner.
     */
    inline void setMinCorner(const PointType &m);

    /*! This method sets mMaxCorner.
     *  \param m Sets mMaxCorner.
     */
    inline void setMaxCorner(const PointType &m);

    /*! A AxisAlignedHypercube keeps track of the coordinates of the minimal corner.
     */
    PointType mMinCorner;

    /*! A AxisAlignedHypercube keeps track of the coordinates of the maximal corner.
     */
    PointType mMaxCorner;
}; // end template class AxisAlignedHypercube

#include "AxisAlignedHypercube.inl"

#endif // AXIS_ALIGNED_HYPERCUBE_H

