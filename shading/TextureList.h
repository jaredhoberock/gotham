/*! \file TextureList.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a list of Textures.
 */

#pragma once

#include <vector>
#include <boost/shared_ptr.hpp>
#include "Texture.h"
#include "../include/TextureHandle.h"

class TextureList
  : public std::vector<boost::shared_ptr<Texture> >
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef std::vector<boost::shared_ptr<Texture> > Parent;

    /*! Null destructor does nothing.
     */
    virtual ~TextureList(void);
}; // end TextureList;

