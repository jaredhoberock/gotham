/*! \file FunctionAllocator.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a 
 *         singleton class for allocating
 *         lots of small types.
 */

#ifndef FUNCTION_ALLOCATOR_H
#define FUNCTION_ALLOCATOR_H

class FunctionAllocator
{
  public:
    /*! This method allocates memory for a new
     *  shading function object.
     *  \return A pointer to a newly allocated chunk of memory
     *          large enough to accomodate a shading function.
     */
    static void *malloc(void);

    /*! This method deallocates all the currently
     *  allocated memory.
     */
    static void freeAll(void);
}; // end FunctionAllocator

#endif // FUNCTION_ALLOCATOR_H

