/*!
 * Rectangular Pyramid Partitioning using Integrated Depth Sensors
 *
 * Copyright 2020 by Nathan Bucki <nathan_bucki@berkeley.edu>
 *
 * This code is free software: you can redistribute
 * it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This code is distributed in the hope that it will
 * be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with the code.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "CommonMath/Vec3.hpp"

namespace RectangularPyramidPlanner {

class Pyramid {
 public:

  //! Default constructor. Not a valid pyramid.
  Pyramid()
      : depth(std::numeric_limits<double>::quiet_NaN()),
        rightPixBound(std::numeric_limits<int>::quiet_NaN()),
        topPixBound(std::numeric_limits<int>::quiet_NaN()),
        leftPixBound(std::numeric_limits<int>::quiet_NaN()),
        bottomPixBound(std::numeric_limits<int>::quiet_NaN()) {
  }

  //! Creates a new pyramid.
  /*!
   * @param depthIn The depth of the base plane (perpendicular to the z-axis) in meters
   * @param edgesIn The position in pixel coordinates of each lateral face of the pyramid.
   * Because we're using the pinhole camera model, when the pyramid is projected into the image
   * plane, it appears as a rectangle. The entries of the array (in order) are the right, top, left,
   * and bottom edges of the rectangle (where right > left and bottom > top).
   * @param corners The position of the corners of the pyramid as written in the camera-fixed frame.
   * Each corner should have the same depth. [meters]
   */
  Pyramid(double depthIn, int edgesIn[4], CommonMath::Vec3 corners[4])
      : depth(depthIn),
        rightPixBound(edgesIn[0]),
        topPixBound(edgesIn[1]),
        leftPixBound(edgesIn[2]),
        bottomPixBound(edgesIn[3]) {
    planeNormals[0] = corners[0].Cross(corners[1]).GetUnitVector();
    planeNormals[1] = corners[1].Cross(corners[2]).GetUnitVector();
    planeNormals[2] = corners[2].Cross(corners[3]).GetUnitVector();
    planeNormals[3] = corners[3].Cross(corners[0]).GetUnitVector();
  }

  //! We define this operator so that we can sort pyramid by the depth of their base planes
  bool operator<(const Pyramid& rhs) const {
    return depth < rhs.depth;
  }

  //! We define this operator so that we can search a sorted list of pyramids and ignore those
  //! at a shallower depth that a given sample point.
  bool operator<(const double rhs) const {
    return depth < rhs;
  }

  //! Depth of the base plane of the pyramid [meters]
  double depth;
  //! Location of the right lateral face projected into the image, making it a line located between leftPixBound and the image width
  int rightPixBound;
  //! Location of the top lateral face projected into the image, making it a line located between 0 and bottomPixBound
  int topPixBound;
  //! Location of the left lateral face projected into the image, making it a line located between 0 and rightPixBound
  int leftPixBound;
  //! Location of the bottom lateral face projected into the image, making it a line located between topPixBound and the image height
  int bottomPixBound;
  //! Unit normals of the lateral faces of the pyramid
  CommonMath::Vec3 planeNormals[4];
};
}  // namespace RectangularPyramidPlanner
