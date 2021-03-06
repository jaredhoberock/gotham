/*! \file SimpleBidirectionalSampler.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a PathSampler
 *         performing a simplified Veach-style
 *         bidirectional sampling algorithm.
 */

#ifndef SIMPLE_BIDIRECTIONAL_SAMPLER_H
#define SIMPLE_BIDIRECTIONAL_SAMPLER_H

#include "PathSampler.h"
#include <aliastable/AliasTable.h>
#include <vector>

class SimpleBidirectionalSampler
  : public PathSampler
{
  public:
    SimpleBidirectionalSampler(void);

    /*! Constructor accepts a maximum path length.
     *  \param maxLength Sets the maximum length for paths
     *                   created by this SimpleBidirectionalSampler.
     */
    SimpleBidirectionalSampler(const unsigned int maxLength);

    /*! This method initializes this SimpleBidirectionalSampler for
     *  sampling.
     *  \param maxLength Sets the maximum length for paths
     *                   created by this SimpleBidirectionalSampler.
     */
    virtual void init(const unsigned int maxLength);

    /*! This method constructs a Path given a
     *  HyperPoint uniquely specifying a Path in a
     *  Scene of interest.
     *  \param scene The Scene containing the environment to
     *               construct a Path in.
     *  \param context A ShadingContext for evaluating shaders.
     *  \param x A HyperPoint uniquely specifying the Path to
     *           construct.
     *  \param p The constructed Path will be returned here.
     *  \return true if a Path could be constructed; false, otherwise.
     */
    virtual bool constructPath(const Scene &scene,
                               ShadingContext &context,
                               const HyperPoint &x,
                               Path &p);

    /*! This method evaluates this Path's Monte Carlo contribution.
     *  \param scene The Scene containing the environment to construct
     *               a Path in.
     *  \param p The Path of interest assumed to be constructed by this
     *           SimpleBidirectionalSampler.
     *  \param results A list of Monte Carlo contributions, binned by
     *                 subpath length, is returned here.
     */
    virtual void evaluate(const Scene &scene,
                          const Path &p,
                          std::vector<Result> &results) const;

  protected:
    /*! This method constructs the eye subpath of a given Path.
     *  \param scene The Scene containing the environment to
     *               construct a Path in.
     *  \param context A ShadingContext for evaluating shaders.
     *  \param x A HyperPoint uniquely specifying the Path to
     *           construct.
     *  \param numVertices The number of vertices to include
     *                     in the eye subpath.
     *  \param p This Path's eye subpath will be updated to
     *           reflect a new subpath.
     *  \return true if an eye subpath could be constructed;
     *          false, otherwise.
     */
    virtual bool constructEyePath(const Scene &scene,
                                  ShadingContext &context,
                                  const HyperPoint &x,
                                  const size_t numVertices,
                                  Path &p) const;

    /*! This method constructs the light subpath of a given Path.
     *  \param scene The Scene containing the environment to
     *               construct a Path in.
     *  \param context A ShadingContext for evaluating shaders.
     *  \param x A HyperPoint uniquely specifying the Path to
     *           construct.
     *  \param numVertices The number of vertices to include
     *                     in the light subpath.
     *  \param p This Path's light subpath will be updated to
     *           reflect a new light subpath.
     *  \return true if an light subpath could be constructed;
     *          false, otherwise.
     */
    virtual bool constructLightPath(const Scene &scene,
                                    ShadingContext &context,
                                    const HyperPoint &x,
                                    const size_t numVertices,
                                    Path &p) const;
                                  
    /*! This method constructs mPathLengthPdf
     *  and mSubpathLengthPdfs.
     */
    virtual void constructPdfs(void);

    /*! This method evaluates the area product measure pdf of the given Path
     *  and subpath lengths.
     *  \param scene The Scene containing the Path of interest.
     *  \param p The Path of interest.
     *  \param s The length of the light subpath of interest.
     *  \param t The length of the eye subpath of interest.
     *  \return The area product measure of choosing p from s light vertices
     *          and t eye vertices.
     */
    float computeAreaProductMeasurePdf(const Scene &scene,
                                       const Path &p,
                                       const size_t s,
                                       const size_t t) const;

    /*! This method computes the MIS weight of a Path generated by this
     *  SimpleBidirectionalSampler.
     *  \param scene The Scene containing p.
     *  \param p The Path of interest.
     *  \return The MIS weight of p.
     */
    float computeMultipleImportanceSamplingWeight(const Scene &scene,
                                                  const Path &p) const;

    AliasTable<size_t, float> mPathLengthPdf;

    unsigned int mMinEyeSubpathLength;
    unsigned int mMinLightSubpathLength;

    unsigned int mMaxEyeSubpathLength;
    unsigned int mMaxLightSubpathLength;

    unsigned int mMinPathLength;
    unsigned int mMaxPathLength;

    typedef AliasTable<std::pair<size_t, size_t>, float> SubpathLengthPdf;
    std::vector<SubpathLengthPdf> mSubpathLengthPdfs;
}; // end SimpleBidirectionalSampler

#endif // SIMPLE_BIDIRECTIONAL_SAMPLER_H

