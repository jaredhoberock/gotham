/*! \file MaterialList.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a list of Materials.
 */

#pragma once

#include <vector>
#include <boost/shared_ptr.hpp>
#include "../include/detail/Material.h"
#include "MaterialHandle.h"

class MaterialList
  : public std::vector<boost::shared_ptr<Material> >
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef std::vector<boost::shared_ptr<Material> > Parent;

    /*! Null destructor does nothing.
     */
    virtual ~MaterialList(void);
}; // end MaterialList

