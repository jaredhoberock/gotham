#include <boost/python.hpp>
using namespace boost::python;

char const *greet()
{
  return "hello, world";
}

BOOST_PYTHON_MODULE(hello)
{
  def("greet", greet);
}

