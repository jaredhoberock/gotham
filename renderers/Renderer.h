/*! \file Renderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class for performing the
 *         high-level operations of rendering.
 */

#ifndef RENDERER_H
#define RENDERER_H

class Scene;
class Record;
class ShadingContext;

#ifdef WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#endif // WIN32

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/timer.hpp>
#include <boost/progress.hpp>

/*! \class Renderer
 *  \brief A Renderer abstracts the high-level
 *         coordination of rendering tasks.
 *         Renderer is an abstract class: an
 *         implementation must implement kernel().
 */
class Renderer
{
  public:
    /*! \typedef ProgressCallback
     *  \brief A ProgressCallback is a callback
     *         to call every so often.
     */
    class ProgressCallback
      : public boost::progress_display
    {
      public:
        /*! \typedef Parent
         *  \brief Shorthand.
         */
        typedef boost::progress_display Parent;
        ProgressCallback(void);
        virtual void restart(unsigned long expected_count);
        virtual ~ProgressCallback(void){;}
    }; // end ProgressCallback

    /*! Null constructor nulls mScene and mCamera.
     */
    inline Renderer(void);

    /*! Constructor accepts a pointer to a Scene and Record.
     *  \param s Sets mScene.
     *  \param r Sets mRecord.
     */
    inline Renderer(boost::shared_ptr<const Scene>  s,
                    boost::shared_ptr<Record> r);

    /*! Null destructor does nothing.
     */
    inline virtual ~Renderer(void);

    /*! This method renders mScene to mFilm.
     *  \param progress A callback, which will be periodically
     *         called throughout the rendering process.
     */
    virtual void render(ProgressCallback &progress);

    /*! This method sets mScene.
     *  \param s Sets mScene.
     */
    inline virtual void setScene(const boost::shared_ptr<const Scene> &s);

    /*! This method returns mScene.
     *  \return mScene.
     */
    inline boost::shared_ptr<const Scene> getScene(void) const;

    /*! This method sets mRecord.
     *  \param r Sets mRecord.
     */
    inline void setRecord(boost::shared_ptr<Record> f);

    /*! This method sets mShadingContext.
     *  \param s Sets mShadingContext.
     */
    inline virtual void setShadingContext(const boost::shared_ptr<ShadingContext> &s);

    /*! This method returns a const reference to mShadingContext.
     *  \return mShadingContext.
     */
    inline const ShadingContext &getShadingContext(void) const;

    /*! This method returns a reference to mShadingContext.
     *  \return mShadingContext.
     */
    inline ShadingContext &getShadingContext(void);

    /*! This method returns mRecord.
     *  \return mRecord.
     */
    inline boost::shared_ptr<const Record> getRecord(void) const;

    /*! This virtual method returns a string describing the parameters
     *  of the render.
     *  \return A string describing the render.
     *  \note Returns a null string by default.
     */
    virtual std::string getRenderParameters(void) const;

  protected:
    /*! This method is called before kernel().
     */
    virtual void preprocess(void);

    /*! This method should be implemented in a derived class to
     *  perform the actual rendering task.
     *  \param progress A callback, which will be periodically
     *         called throughout the rendering process.
     */
    virtual void kernel(ProgressCallback &progress) = 0;

    /*! This method is called after kernel().
     */
    virtual void postprocess(void);

    /*! This method reports render statistics.
     *  \param elapsed The length of the render, in seconds.
     */
    virtual void postRenderReport(const double elapsed) const;

    /*! A Renderer keeps a pointer to the Scene currently being rendered.
     */
    boost::shared_ptr<const Scene> mScene;

    /*! A Renderer keeps a pointer to the RandomAccessFilm to which mScene is rendered.
     */
    boost::shared_ptr<Record> mRecord;

    /*! A Renderer keeps a pointer to a ShadingContext for evaluating shaders.
     */
    boost::shared_ptr<ShadingContext> mShadingContext;

    /*! A timer.
     */
    boost::timer mTimer;
}; // end Renderer

#include "Renderer.inl"

#endif // RENDERER_H

