/*! \file UnshadowedScene.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Scene
 *         in which shadow rays are never blocked.
 */

#ifndef UNSHADOWED_SCENE_H
#define UNSHADOWED_SCENE_H

#include "Scene.h"

class UnshadowedScene
  : public Scene
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Scene Parent;

    /*! This method, which evaluates shadow Rays,
     *  always returns false.
     *  \param r The shadow Ray of interest.
     *  \return false
     */
    using Parent::intersect;
    inline virtual bool intersect(const Ray &r) const;
}; // end UnshadowedScene

#include "UnshadowedScene.inl"

#endif // UNSHADOWED_SCENE_H
