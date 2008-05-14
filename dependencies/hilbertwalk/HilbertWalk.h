/*! \file HilbertWalk.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class abstracting
 *         a walk through a Hilbert curve on an integer grid.
 */

#ifndef HILBERT_WALK_H
#define HILBERT_WALK_H

#include <vector>

class HilbertWalk
{
  public:
    inline HilbertWalk(void);

    inline HilbertWalk(const size_t n,
                       const size_t x0,
                       const size_t y0);

    inline void init(const size_t n,
                     const size_t x0,
                     const size_t y0);

    /*! This method returns the next location in the walk.
     *  \param x The x-coordinate is returned here.
     *  \param y The y-coordinate is returned here.
     *  \return true, if the next location could be generated;
     *          false, if the walk is over.
     */
    inline bool operator()(size_t &x, size_t &y);

  protected:
    typedef enum {F, P, M, L, R} TurtleCommand;

    inline void f(void);
    inline void p(void);
    inline void m(void);
    inline void l(const size_t n);
    inline void r(const size_t n);

    /*! The stack of work to do.
     */
    std::vector<std::pair<TurtleCommand, size_t> > mStack;

    /*! The current turtle walk direction.
     */
    gpcpu::int2 mWalkDirection;

    /*! The current turtle position.
     */
    gpcpu::size2 mCurrentPosition;
}; // end HilbertWalk

#include "HilbertWalk.inl"

#endif // HILBERT_WALK_H

