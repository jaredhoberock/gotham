/*! \file SIMDDebugRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a SIMD renderer which implements the
 *         same functionality as DebugRenderer.
 */

#ifndef SIMD_DEBUG_RENDERER_H
#define SIMD_DEBUG_RENDERER_H

#include "SIMDRenderer.h"

class SIMDDebugRenderer
  : public SIMDRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef SIMDRenderer Parent;

    /*! Null constructor calls the Parent
     */
    inline SIMDDebugRenderer(void);

    /*! Constructor accepts a pointer to a Scene and Record.
     *  \param s Sets mScene.
     *  \param r Sets mRecord.
     */
    inline SIMDDebugRenderer(boost::shared_ptr<const Scene>  s,
                             boost::shared_ptr<Record> r);

  protected:
    /*! This method performs kernel work for a single thread.
     *  \param threadIdx The thread index.
     */
    virtual void kernel(const size_t threadIdx);
}; // end SIMDDebugRenderer

#include "SIMDDebugRenderer.inl"

#endif // SIMD_DEBUG_RENDERER_H

