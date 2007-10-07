/*! \file AxisAlignedHypercube.inl
 *  \author Jared Hoberock
 *  \brief Inline file for AxisAlignedHypercube.h.
 */

#include <limits>

template<typename PointType, typename RealType, unsigned int mNumDimensions>
  AxisAlignedHypercube<PointType, RealType, mNumDimensions>::AxisAlignedHypercube(void)
{
  setEmpty();
} // end AxisAlignedHypercube::AxisAlignedHypercube()

template<typename PointType, typename RealType, unsigned int mNumDimensions>
  void AxisAlignedHypercube<PointType, RealType, mNumDimensions>::setEmpty(void)
{
  PointType inf;
  for(int i = 0; i < mNumDimensions; ++i)
  {
    inf[i] = std::numeric_limits<RealType>::infinity();
  } // end for i

  setMinCorner(inf);
  setMaxCorner(-inf);
} // end AxisAlignedHypercube::setEmpty()

template<typename PointType, typename RealType, unsigned int mNumDimensions>
  AxisAlignedHypercube<PointType, RealType, mNumDimensions>::AxisAlignedHypercube(const PointType &min, const PointType &max)
{
  set(min,max);
} // end AxisAlignedHypercube::AxisAlignedHypercube()

template<typename PointType, typename RealType, unsigned int mNumDimensions>
  void AxisAlignedHypercube<PointType, RealType, mNumDimensions>::set(const PointType &min, const PointType &max)
{
  setMinCorner(min);
  setMaxCorner(max);
} // end AxisAlignedHypercube::set()

template<typename PointType, typename RealType, unsigned int mNumDimensions>
  void AxisAlignedHypercube<PointType, RealType, mNumDimensions>::setMinCorner(const PointType &m)
{
  mMinCorner = m;
} // end AxisAlignedHypercube::setMinCorner()

template<typename PointType, typename RealType, unsigned int mNumDimensions>
  void AxisAlignedHypercube<PointType, RealType, mNumDimensions>::setMaxCorner(const PointType &m)
{
  mMaxCorner = m;
} // end AxisAlignedHypercube::setMaxCorner()

template<typename PointType, typename RealType, unsigned int mNumDimensions>
  const PointType &AxisAlignedHypercube<PointType, RealType, mNumDimensions>::getMinCorner(void) const
{
  return mMinCorner;
} // end AxisAlignedHypercube::getMinCorner()

template<typename PointType, typename RealType, unsigned int mNumDimensions>
  const PointType &AxisAlignedHypercube<PointType, RealType, mNumDimensions>::getMaxCorner(void) const
{
  return mMaxCorner;
} // end AxisAlignedHypercube::getMaxCorner()

template<typename PointType, typename RealType, unsigned int mNumDimensions>
  PointType &AxisAlignedHypercube<PointType, RealType, mNumDimensions>
    ::getMinCorner(void)
{
  return mMinCorner;
} // end AxisAlignedHypercube::getMinCorner()

template<typename PointType, typename RealType, unsigned int mNumDimensions>
  PointType &AxisAlignedHypercube<PointType, RealType, mNumDimensions>
    ::getMaxCorner(void)
{
  return mMaxCorner;
} // end AxisAlignedHypercube::getMaxCorner()

template<typename PointType, typename RealType, unsigned int mNumDimensions>
  bool AxisAlignedHypercube<PointType, RealType, mNumDimensions>
    ::addPoint(const PointType &p)
{
  bool result = false;

  // for each dimension
  for(int i = 0; i < mNumDimensions; ++i)
  {
    if(p[i] < getMinCorner()[i])
    {
      mMinCorner[i] = p[i];
      result = true;
    } // end if
    if(p[i] > getMaxCorner()[i])
    {
      mMaxCorner[i] = p[i];
      result = true;
    } // end else if
  } // end for i

  return result;
} // end AxisAlignedHypercube::addPoint()

template<typename PointType, typename RealType, unsigned int mNumDimensions>
  PointType &AxisAlignedHypercube<PointType, RealType, mNumDimensions>::operator[](const unsigned int i)
{
  if(i == 0) return getMinCorner();

  return getMaxCorner();
} // end AxisAlignedHypercube::operator[]()

template<typename PointType, typename RealType, unsigned int mNumDimensions>
  const PointType &AxisAlignedHypercube<PointType, RealType, mNumDimensions>::operator[](const unsigned int i) const
{
  if(i == 0) return getMinCorner();

  return getMaxCorner();
} // end AxisAlignedHypercube::operator[]()

template<typename PointType, typename RealType, unsigned int mNumDimensions>
  RealType AxisAlignedHypercube<PointType, RealType, mNumDimensions>::computeDiagonalLength(void) const
{
  return (getMaxCorner() - getMinCorner()).length();
} // end AxisAlignedHypercube::computeDiagonalLength()

template<typename PointType, typename RealType, unsigned int mNumDimensions>
  RealType AxisAlignedHypercube<PointType, RealType, mNumDimensions>::computeVolume(void) const
{
  RealType result = 1.0;

  for(unsigned int dimension = 0; dimension < mNumDimensions; ++dimension)
  {
    // multiply by the length of each dimension
    result *= getMaxCorner()[dimension] - getMinCorner()[dimension];
  } // end for dimension

  return result;
} // end AxisAlignedHypercube::computeVolume()

template<typename PointType, typename RealType, unsigned mNumDimensions>
  PointType AxisAlignedHypercube<PointType, RealType, mNumDimensions>::computeCentroid(void) const
{
  return (getMinCorner() + getMaxCorner()) / 2.0;
} // end AxisAlignedHypercube::computeCentroid()

