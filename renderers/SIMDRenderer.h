/*! \file SIMDRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a single instruction multiple
 *         data style parallel Renderer.
 */

#ifndef SIMD_RENDERER_H
#define SIMD_RENDERER_H

#include "Renderer.h"

class SIMDRenderer
  : public Renderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Renderer Parent;

    /*! Null constructor calls the Parent.
     */
    inline SIMDRenderer(void);

    /*! Constructor accepts a pointer to a Scene and Record.
     *  \param s Sets mScene.
     *  \param r Sets mRecord.
     */
    inline SIMDRenderer(boost::shared_ptr<const Scene>  s,
                        boost::shared_ptr<Record> r);

    /*! This method sets the number of threads to use.
     *  \param n Sets mWorkBatchSize
     */
    inline void setWorkBatchSize(const size_t n);

  protected:
    /*! This method renders mScene to mFilm.
     *  \param progress A callback, which will be periodically
     *         called throughout the rendering process.
     */
    virtual void kernel(ProgressCallback &progress);

    /*! This method performs kernel work for a single thread.
     *  \param threadIdx The thread index.
     */
    virtual void kernel(const size_t threadIdx) = 0;

    /*! This parameter controls the batch size of the workload.
     */
    size_t mWorkBatchSize;
}; // end SIMDRenderer

#include "SIMDRenderer.inl"

#endif // SIMD_RENDERER_H

