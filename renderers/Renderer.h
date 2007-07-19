/*! \file Renderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class for performing the
 *         high-level operations of rendering.
 */

#ifndef RENDERER_H
#define RENDERER_H

class Scene;
class RandomAccessFilm;

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

/*! \class Renderer
 *  \brief A Renderer abstracts the high-level
 *         coordination of rendering tasks.
 *         Renderer is an abstract class: an
 *         implementation must implement render().
 */
class Renderer
{
  public:
    /*! \typedef ProgressCallback
     *  \brief A ProgressCallback is a callback
     *         to call every so often.
     */
    typedef boost::function1<void,unsigned int> ProgressCallback;

    /*! Null constructor nulls mScene and mCamera.
     */
    inline Renderer(void);

    /*! Constructor accepts a pointer to a Scene, Camera, and Film.
     *  \param s Sets mScene.
     *  \param f Sets mFilm.
     */
    inline Renderer(boost::shared_ptr<const Scene>  s,
                    boost::shared_ptr<RandomAccessFilm> f);

    /*! This method renders mScene to mFilm.
     *  \param progress A callback, which will be periodically
     *         called throughout the rendering process.
     */
    virtual void render(ProgressCallback &progress) = 0;

    /*! This method sets mScene.
     *  \param s Sets mScene.
     */
    inline void setScene(boost::shared_ptr<const Scene> s);

    /*! This method returns mScene.
     *  \return mScene.
     */
    inline boost::shared_ptr<const Scene> getScene(void) const;

    /*! This method sets mFilm.
     *  \param f Sets mFilm.
     */
    inline void setFilm(boost::shared_ptr<RandomAccessFilm> f);

    /*! This method returns mFilm.
     *  \return mFilm.
     */
    inline boost::shared_ptr<const RandomAccessFilm> getFilm(void) const;

  protected:
    /*! A Renderer keeps a pointer to the Scene currently being rendered.
     */
    boost::shared_ptr<const Scene> mScene;

    /*! A Renderer keeps a pointer to the RandomAccessFilm to which mScene is rendered.
     */
    boost::shared_ptr<RandomAccessFilm> mFilm;
}; // end Renderer

#include "Renderer.inl"

#endif // RENDERER_H

