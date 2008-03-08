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
    /*! This parameter controls the batch size of the workload.
     */
    size_t mWorkBatchSize;
}; // end SIMDRenderer

#include "SIMDRenderer.inl"

#endif // SIMD_RENDERER_H

