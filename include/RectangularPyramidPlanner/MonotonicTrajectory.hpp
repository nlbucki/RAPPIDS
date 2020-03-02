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
#include "CommonMath/Trajectory.hpp"

class MonotonicTrajectory : public CommonMath::Trajectory {
 public:

  //! Creates a trajectory with monotonically changing depth (i.e. position along the z-axis).
  //! Note that we do not check if the given trajectory actually has monotonically z-position
  //! between the start and end times; we assume this is the case.
  /*!
   * @param coeffs The parameters defining the trajectory
   * @param startTime Endpoint of the trajectory [seconds]
   * @param endTime Endpoint of the trajectory [seconds]
   */
  MonotonicTrajectory(std::vector<CommonMath::Vec3> coeffs, double startTime,
                      double endTime)
      : CommonMath::Trajectory(coeffs, startTime, endTime) {
    double startVal = Trajectory::GetAxisValue(2, startTime);
    double endVal = Trajectory::GetAxisValue(2, endTime);
    increasingDepth = startVal < endVal;  // index 2 = z-value
  }

  //! We include this operator so that we can sort the monotonic sections based on the depth
  //! of their deepest point. The idea is that we should check the monotonic section with the
  //! deepest depth for collisions first, as it's the most likely to collide with the environment.
  bool operator<(const MonotonicTrajectory& rhs) const {
    double deepestDepth, rhsDeepestDepth;
    if (increasingDepth) {
      deepestDepth = Trajectory::GetAxisValue(2, Trajectory::GetEndTime());
    } else {
      deepestDepth = Trajectory::GetAxisValue(2, Trajectory::GetStartTime());
    }
    if (rhs.increasingDepth) {
      rhsDeepestDepth = rhs.GetAxisValue(2, rhs.GetEndTime());
    } else {
      rhsDeepestDepth = rhs.GetAxisValue(2, rhs.GetStartTime());
    }
    return deepestDepth < rhsDeepestDepth;
  }

  //! True if the position of the trajectory along the z-axis monotonically increases in time
  bool increasingDepth;
};
