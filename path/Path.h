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
#include "../shading/ScatteringDistributionFunction.h"
#include <spectrum/Spectrum.h>
#include <boost/array.hpp>

class FunctionAllocator;
class SurfacePrimitive;
class SurfacePrimitiveList;
class Scene;
class RussianRoulette;

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
   *  \param surface Sets mSurface.
   *  \param dg Sets mDg.
   *  \param emission Sets mEmission.
   *  \param scattering Sets mScattering.
   *  \param sensor Sets mSensor.
   *  \param f Sets mThroughput.
   *  \param accumulatedPdf Sets mAccumulatedPdf.
   *  \param pdf Sets mPdf.
   *  \param delta Sets mFromDelta.
   *  \param component Sets mFromComponent.
   */
  void init(const SurfacePrimitive *surface,
            const DifferentialGeometry &dg,
            const ScatteringDistributionFunction *emission,
            const ScatteringDistributionFunction *integrand,
            const ScatteringDistributionFunction *sensor,
            const Spectrum &f,
            const float accumulatedPdf,
            const float pdf,
            const bool delta,
            const ScatteringDistributionFunction::ComponentIndex component);

  // DifferentialGeometry describes the intersection point
  DifferentialGeometry mDg;

  // The emission function at the intersection point:
  const ScatteringDistributionFunction *mEmission;

  // The scattering function at the intersection point:
  const ScatteringDistributionFunction *mScattering;

  // The sensor function at the intersection point:
  const ScatteringDistributionFunction *mSensor;

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

  // true if this PathVertex was sampled from a Dirac
  // delta distribution; false, otherwise.
  bool mFromDelta;

  // this is set to the index of the component this PathVertex
  // was sampled from the previous ScatteringDistributionFunction
  ScatteringDistributionFunction::ComponentIndex mFromComponent;
}; // end PathVertex

#define PATH_LENGTH 20

class Path
  : public boost::array<PathVertex, PATH_LENGTH>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef boost::array<PathVertex, PATH_LENGTH> Parent;

    static const unsigned int ROULETTE_TERMINATED = UINT_MAX;
    static const unsigned int INSERT_FAILED = ROULETTE_TERMINATED - 1;
    static const unsigned int INSERT_SUCCESS = INSERT_FAILED - 1;

    /*! Null constructor does nothing.
     */
    Path(void);

    /*! This method clears this Path.
     */
    void clear(void);

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
     *  \return i if a new PathVertex could be inserted; INSERT_FAILED, otherwise.
     */
    unsigned int insert(const unsigned int i,
                        const SurfacePrimitiveList *surfaces,
                        const bool emission,
                        const float u0,
                        const float u1,
                        const float u2,
                        const float u3);

    /*! This method:
     *  - inserts a PathVertex at the given index by sampling from a SurfacePrimitive.
     *  \param i The index at which to insert the PathVertex.
     *  \param surfaces The SurfacePrimitiveList to sample from.
     *  \param emission If this is true, the new PathVertex is initialized
     *                  with an EmissionFunction; otherwise, a SensorFunction.
     *  \param u0 A real number in [0,1).
     *  \param u1 A real number in [0,1).
     *  \param u2 A real number in [0,1).
     *  \note This method assumes that the new PathVertex begins a new
     *        eye or light path, depending on the value of emission.
     *  \return i if a new PathVertex could be inserted; INSERT_FAILED, otherwise.
     */
    unsigned int insert(const unsigned int i,
                        const SurfacePrimitive *prim,
                        const bool emission,
                        const float u0,
                        const float u1,
                        const float u2);

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
     *          is found; INSERT_FAILED, otherwise.
     */
    unsigned int insert(const unsigned int previous,
                        const Scene *scene,
                        const bool after,
                        const bool scatter,
                        const float u0,
                        const float u1,
                        const float u2);

    /*! This method:
     *  - inserts a PathVertex before or after the given index by
     *    sampling the previous PathVertex's integrand and applying Russian roulette.
     *  \param previous The index of the previous vertex.
     *  \param scene A pointer to the Scene containing this Path.
     *  \param after Whether or not to insert the new PathVertex before
     *               or after previous.
     *  \param scatter Whether or not to perform bidirectional scattering
     *                 or unidirectional emission.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param u3 A fourth real number in [0,1).
     *  \param roulette A Russian roulette function for computing whether or not
     *                  to extend the Path.
     *  \return The index of the newly inserted PathVertex if an intersection
     *          is found; ROULETTE_TERMINATED, if roulette kills the insertion, or
     *          INSERT_FAILED if no new PathVertex could be found.
     */
    unsigned int insertRussianRoulette(const unsigned int previous,
                                       const Scene *scene,
                                       const bool after,
                                       const bool scatter,
                                       const float u0,
                                       const float u1,
                                       const float u2,
                                       const float u3,
                                       const RussianRoulette *roulette);
    
    /*! This variant of the insert() method attempts to append (or prepend) a new PathVertex
     *  using Russian roulette.  The termination probability is also returned.
     *  \param previous The index of the previous vertex.
     *  \param scene A pointer to the Scene containing this Path.
     *  \param after Whether or not to insert the new PathVertex before
     *               or after previous.
     *  \param scatter Whether or not to perform bidirectional scattering
     *                 or unidirectional emission.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param u3 A fourth real number in [0,1).
     *  \param roulette A Russian roulette function for computing whether or not
     *                  to extend the Path.
     *  \param termination The probability of terminating this Path due to Russian roulette
     *                     is returned here.
     *  \return The index of the newly inserted PathVertex if an intersection
     *          is found; ROULETTE_TERMINATED, if roulette kills the insertion, or
     *          INSERT_FAILED if no new PathVertex could be found.
     */
    unsigned int insertRussianRouletteWithTermination(const unsigned int previous,
                                                      const Scene *scene,
                                                      const bool after,
                                                      const bool scatter,
                                                      const float u0,
                                                      const float u1,
                                                      const float u2,
                                                      const float u3,
                                                      const RussianRoulette *roulette,
                                                      float &termination);


    template<typename RNG>
      bool construct(const Scene *scene,
                     const unsigned int eyeSubpathLength,
                     const unsigned int lightSubpathLength,
                     const float p0,
                     const float p1,
                     const float p2,
                     RNG &rng);

    Spectrum computeThroughputAssumeVisibility(const unsigned int eyeSubpathLength,
                                               const PathVertex &e,
                                               const unsigned int lightSubpathLength,
                                               const PathVertex &l) const;
    
    void connect(PathVertex &v0,
                 PathVertex &v1);

    const gpcpu::uint2 &getSubpathLengths(void) const;

    /*! This method returns a reference to this Path's termination probabilities.
     *  \return mTerminationProbabilities
     */
    gpcpu::float2 &getTerminationProbabilities(void);

    /*! This method returns a const reference to this Path's termination probabilities.
     *  \return mTerminationProbabilities
     */
    const gpcpu::float2 &getTerminationProbabilities(void) const;


    // XXX this probably belonds somewhere else
    static float computeG(const Normal &n0,
                          const Vector &w,
                          const Normal &n1,
                          const float d2);

    /*! This method does a safe copy of this Path into a destination
     *  using a FunctionAllocator.
     *  \param dst The destination Path to clone into.
     *  \param allocator A FunctionAllocator for allocating space for ScatteringDistributionFunctions.
     *  \return true if this Path could be successfully cloned; false, if an allocation failed.
     */
    bool clone(Path &dst, FunctionAllocator &allocator) const;

    /*! This static method computes a MIS weight for a Path with the power heuristic.
     *  \param scene The Scene containing the Path.
     *  \param minimumLightSubpathLength The minimum length of a light subpath to consider.
     *  \param minimumEyeSubpathLength The minimum length of an eye subpath to consider.
     *  \param lLast The last vertex of the light subpath to consider.
     *  \param s The length of the light subpath being considered.
     *  \param lightSubpathLength The length of the full light subpath that was generated
     *                            when the Path was sampled.
     *  \param eLast The last vertex of the eye subpath to consider.
     *  \param t The length of the eye subpath being considered.
     *  \param eyeSubpathLength The length of the full eye subpath that was generated
     *                          when the Path was sampled.
     *  \param connection A unit vector pointing from eLast to lLast.
     *  \param g The geometric term between eLast and lLast.
     *  \param roulette The RussianRoulette function used when sampling the Path being
     *                  considered.
     *  \return The power heuristic (with exponent == 2) weight of the Path being considered.
     */
    static float computePowerHeuristicWeight(const Scene &scene,
                                             const size_t minimumLightSubpathLength,
                                             const size_t minimumEyeSubpathLength,
                                             const const_iterator &lLast,
                                             const size_t s,
                                             const size_t lightSubpathLength,
                                             const const_iterator &eLast,
                                             const size_t t,
                                             const size_t eyeSubpathLength,
                                             const Vector &connection,
                                             const float g,
                                             const RussianRoulette &roulette);


    /*! These methods are too confusing to describe.  Don't call them unless you wrote them.
     */
    static float computePowerHeuristicWeightEyeSubpaths(const Scene &scene,
                                                        const size_t minimumLightSubpathLength,
                                                        const Path::const_iterator &lLast,
                                                        const size_t s,
                                                        const size_t lightSubpathLength,
                                                        const Path::const_iterator &eLast,
                                                        const size_t t,
                                                        const size_t eyeSubpathLength,
                                                        const RussianRoulette &roulette);

    static float computePowerHeuristicWeightLightSubpaths(const Scene &scene,
                                                          const size_t minimumEyeSubpathLength,
                                                          const Path::const_iterator &lLast,
                                                          const size_t s,
                                                          const size_t lightSubpathLength,
                                                          const Path::const_iterator &eLast,
                                                          const size_t t,
                                                          const size_t eyeSubpathLength,
                                                          const RussianRoulette &roulette);


  //protected:
    // XXX pass prev as a PathVertex reference
    // XXX dir & pdf need not be passed as parameters:
    //     pass them in the new PathVertex
    unsigned int insert(const unsigned int previous,
                        const Scene *scene,
                        const bool after,
                        const Vector &dir,
                        const Spectrum &f,
                        float pdf,
                        const bool delta,
                        const ScatteringDistributionFunction::ComponentIndex component);

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
                        const bool delta,
                        const ScatteringDistributionFunction::ComponentIndex component,
                        const SurfacePrimitive *surface,
                        const ScatteringDistributionFunction *emission,
                        const ScatteringDistributionFunction *scattering,
                        const ScatteringDistributionFunction *sensor,
                        const DifferentialGeometry &dg);

    // This tuple the length of the eye and
    // light subpaths, respectively.
    gpcpu::uint2 mSubpathLengths;

    // This tuple records the termination probability of each of
    // this Path's eye and light subpaths, respectively.
    // These probabilities correspond to the probability that each
    // subpath was terminated at their final vertices.
    gpcpu::float2 mTerminationProbabilities;

  private:
    /*! Since ScatteringDistributionFunctions are allocated in a weird way,
     *  disallow copies.
     */
    Path(const Path &p);
    Path &operator=(const Path &p);
}; // end Path

#undef PATH_LENGTH

#include "Path.inl"

#endif // PATH_H

