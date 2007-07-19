/*! \file Path.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class
 *         representing a light path.
 */
#ifndef PATH_H
#define PATH_H

#include "../geometry/Point.h"
#include "../geometry/Normal.h"
#include "../geometry/Vector.h"
#include "../geometry/DifferentialGeometry.h"
#include <spectrum/Spectrum.h>
#include <boost/array.hpp>

class ScatteringDistributionFunction;
class SurfacePrimitive;
class SurfacePrimitiveList;
class Scene;

/*! \struct PathVertex
 *  \brief A PathVertex describes surface properties
 *         at a Point along a Path.
 */
struct PathVertex
{
  /*! Null constructor does nothing.
   */
  inline PathVertex(void);

  /*! init method sets the members of this
   *  PathVertex.
   *  \param dg Sets mDg.
   *  \param f Sets mThroughput.
   *  \param accumulatedPdf Sets mAccumulatedPdf.
   *  \param pdf Sets mPdf.
   */
  void init(const DifferentialGeometry &dg,
            const Spectrum &f,
            const float accumulatedPdf,
            const float pdf);

  // DifferentialGeometry describes the intersection point
  DifferentialGeometry mDg;

  // The integrand at the intersection point:
  // either a bidirectional or unidirectional scattering
  // distribution function.
  const ScatteringDistributionFunction *mIntegrand;

  // A pointer to the surface at the intersection point.
  const SurfacePrimitive *mSurface;

  // A unit vector pointing toward the previous
  // PathVertex
  // Here, 'previous' is defined as the adjacent
  // PathVertex with lesser index within the Path.
  Vector mToPrev;

  // a unit vector pointing toward the next
  // PathVertex in the Path
  // Here, 'next' is defined as the adjacent
  // PathVertex with greater index within the Path.
  Vector mToNext;

  // geometric term between this PathVertex
  // and the previous PathVertex
  float mPreviousGeometricTerm;

  // geometric term between this PathVertex
  // and the next PathVertex
  float mNextGeometricTerm;

  // pdf of choosing this PathVertex from the
  // vertex at which it was sampled
  float mPdf;

  // accumulated pdf at this PathVertex
  float mAccumulatedPdf;

  // accumulated throughput at this PathVertex
  // on the Path
  Spectrum mThroughput;
}; // end PathVertex

class Path
  : public boost::array<PathVertex, 6>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef boost::array<PathVertex, 6> Parent;

    static const unsigned int NULL_VERTEX = UINT_MAX;

    /*! This method:
     *  - inserts a PathVertex at the given index by sampling from a
     *    SurfacePrimitive list.
     *  \param i The index at which to insert the PathVertex.
     *  \param surfaces The SurfacePrimitiveList to sample from.
     *  \param emission If this is true, the new PathVertex is initialized
     *                  with an EmissionFunction; otherwise, a SensorFunction.
     *  \param u0 A real number in [0,1).
     *  \param u1 A real number in [0,1).
     *  \param u2 A real number in [0,1).
     *  \param u3 A real number in [0,1).
     *  \note This method assumes that the new PathVertex begins a new
     *        eye or light path, depending on the value of emission.
     */
    void insert(const unsigned int i,
                const SurfacePrimitiveList *surfaces,
                const bool emission,
                const float u0,
                const float u1,
                const float u2,
                const float u3);

    /*! This method:
     *  - inserts a PathVertex before or after the given index by
     *    sampling the previous PathVertex's integrand.
     *  \param previous The index of the previous vertex.
     *  \param scene A pointer to the Scene containing this Path.
     *  \param after Whether or not to insert the new PathVertex before
     *               or after previous.
     *  \param scatter Whether or not to perform bidirectional scattering
     *                 or unidirectional emission.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \return The index of the newly inserted PathVertex if an intersection
     *          is found; NULL_VERTEX, otherwise.
     */
    unsigned int insert(const unsigned int previous,
                        const Scene *scene,
                        const bool after,
                        const bool scatter,
                        const float u0,
                        const float u1,
                        const float u2);

  protected:
    // XXX pass prev as a PathVertex reference
    // XXX dir & pdf need not be passed as parameters:
    //     pass them in the new PathVertex
    unsigned int insert(const unsigned int previous,
                        const Scene *scene,
                        const bool after,
                        const Vector &dir,
                        const Spectrum &f,
                        float pdf);

    // XXX pass prev as a PathVertex reference
    // XXX dir & pdf need not be passed as parameters:
    //     pass them in the new PathVertex
    // XXX dg need not be passed as parameter:
    //     pass it in the new PathVertex
    unsigned int insert(const unsigned int previous,
                        const bool after,
                        const Vector &dir,
                        const Spectrum &f,
                        const float pdf,
                        const DifferentialGeometry &dg);

    // This tuple the length of the eye and
    // light subpaths, respectively.
    gpcpu::uint2 mSubpathLengths;
}; // end Path

#endif // PATH_H

