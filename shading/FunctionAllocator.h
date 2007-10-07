/*! \file FunctionAllocator.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a 
 *         singleton class for allocating
 *         lots of small types.
 */

#ifndef FUNCTION_ALLOCATOR_H
#define FUNCTION_ALLOCATOR_H

#include <vector>
#include "exportShading.h"

// XXX this is nasty: kill it
#ifndef WIN32
#define DLLAPI
#endif // WIN32

class FunctionAllocator
{
  public:
    /*! Null constructor allocates a huge number of
     *  slots in mStorage.
     */
    FunctionAllocator(void);

    /*! Constructor accepts the number of slots to allocate.
     *  \param n The number of slots to allocate.
     */
    FunctionAllocator(const size_t n);

    /*! This method invalidates all Blocks currently allocated
     *  and resizes the number of slots.
     *  \param n The number of slots to allocate.
     */
    void reserve(const size_t n);
    
    /*! This method allocates memory for a new
     *  shading function object.
     *  \return A pointer to a newly allocated chunk of memory
     *          large enough to accomodate a shading function.
     */
    void *malloc(void);

    /*! This method deallocates all the currently
     *  allocated memory.
     */
    void freeAll(void);

    template<unsigned int size>
      struct DLLAPI Accomodator
    {
      unsigned char mFill[size];
    }; // end Accomodator

    typedef Accomodator<64> Block;

  protected:

#ifdef WIN32
    // EXPORT/IMPORT std::vector stuff
    template class DLLAPI std::allocator<Block>;
    template class DLLAPI std::vector<Block>;
#endif // WIN32

    std::vector<Block> mStorage;
}; // end FunctionAllocator

#endif // FUNCTION_ALLOCATOR_H

