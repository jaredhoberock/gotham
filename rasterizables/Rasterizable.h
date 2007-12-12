/*! \file Rasterizable.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an abstract
 *         base class for things that are rasterizable.
 */

#ifndef RASTERIZABLE_H
#define RASTERIZABLE_H

class Rasterizable
{
  public:
    /*! Null destructor does nothing.
     */
    inline virtual ~Rasterizable(void){;};

    /*! This method rasterizes this object.
     */
    virtual void rasterize(void) = 0;
}; // end Rasterizable

#endif // RASTERIZABLE_H

