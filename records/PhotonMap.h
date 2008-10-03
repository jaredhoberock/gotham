/*! \file PhotonMap.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class abstracting a photon map.
 */

#ifndef PHOTON_MAP_H
#define PHOTON_MAP_H

#include <vector>
#include "../include/detail/Point.h"
#include "../include/detail/Vector.h"
#include "../include/detail/Spectrum.h"

struct Photon
{
  Vector mWi;
  Spectrum mPower;
  Point mPoint;
}; // end Photon

class PhotonMap
  : public std::vector<Photon>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef std::vector<Photon> Parent;

    /*! Null constructor does nothing.
     */
    inline PhotonMap(void);

    /*! Constructor accepts a list of Photon positions,
     *  incoming vectors, and powers.
     *  \param positions This list specifies each Photon's position.
     *  \param wi This list specifies each Photon's incoming direction.
     *  \param power This list specified each Photon's power.
     */
    inline PhotonMap(const std::vector<Point> &positions,
                     const std::vector<Vector> &wi,
                     const std::vector<Spectrum> &power);

    /*! This method sorts this PhotonMap.
     */
    inline void sort(void);

    /*! This method performs the nearest neighbors operation.
     *  \param p The Point of interest.
     *  \param maxDist2 The maximum distance squared to a Photon of interest.
     *  \param visitor A visitor functor which will be invoked on nearby Photons.
     */
    template<typename Visitor>
      inline void rangeQuery(const Point &p,
                             float &maxDist2,
                             Visitor &visitor) const;

    /*! This method writes this PhotonMap's data to Python code
     *  which will instantiate it.
     *  \param os The ostream to write to.
     *  \param pm The PhotonMap to write.
     */
    friend inline std::ostream &operator<<(std::ostream &os, const PhotonMap &pm);

  protected:
    template<typename Visitor>
      inline void rangeQuery(const unsigned int b,
                             const unsigned int e,
                             const Point &p,
                             float &maxDist2,
                             Visitor &visitor) const;


    /*! This method sorts the given range of this PhotonMap.
     *  \param b The first element to sort.
     *  \param e The first element beyond the last element to sort.
     */
    inline void sort(const iterator &b, const iterator &e);
    inline void sort(const size_t b, const size_t e);

    /*! This method finds the bounds of the given range of Photons.
     *  \param b The first element to sort.
     *  \param e The first element beyond the last element to sort.
     *  \param m The min bounds of the range is returned here.
     *  \param M The max bounds of the range is returned here.
     */
    inline static void findBounds(const iterator &b, const iterator &e,
                                  Point &m, Point &M);

    struct PhotonSorter
    {
      inline bool operator()(const Photon &lhs, const Photon &rhs) const;
      unsigned int mAxis;
    }; // end PhotonSorter

    // a stack for iterative nearest neighbor search
    typedef std::pair<const_iterator, const_iterator> SearchRange;
    mutable std::vector<SearchRange> mStack;

    struct Node
    {
      float mSplitValue;
      unsigned int mSplitAxis:2;
    }; // end Node

    std::vector<Node> mNodes;
}; // end PhotonMap

#include "PhotonMap.inl"

#endif // PHOTON_MAP_H

