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

#include "RectangularPyramidPlanner/DepthImagePlanner.hpp"

using namespace std::chrono;
using namespace CommonMath;
using namespace RapidQuadrocopterTrajectoryGenerator;
using namespace RectangularPyramidPlanner;

DepthImagePlanner::DepthImagePlanner(cv::Mat depthImage, double depthScale,
                                     double focalLength, double principalPointX,
                                     double principalPointY,
                                     double physicalVehicleRadius,
                                     double vehicleRadiusForPlanning,
                                     double minimumCollisionDistance)
    : _depthScale(depthScale),
      _focalLength(focalLength),
      _cx(principalPointX),
      _cy(principalPointY),
      _imageWidth(depthImage.cols),
      _imageHeight(depthImage.rows),
      _trueVehicleRadius(physicalVehicleRadius),
      _vehicleRadiusForPlanning(vehicleRadiusForPlanning),
      _minCheckingDist(minimumCollisionDistance),
      _minimumAllowedThrust(0),  // By default don't allow negative thrust
      _maximumAllowedThrust(30),  // By default limit maximum thrust to about 3g (30 m/s^2)
      _maximumAllowedAngularVelocity(20),  // By default limit maximum angular velocity to 20 rad/s
      _minimumSectionTimeDynamicFeas(0.02),  // By default restrict the dynamic feasibility check to a minimum section duration of 20ms
      _maxPyramidGenTime(1000),  // Don't limit pyramid generation time by default [seconds].
      _pyramidGenTimeNanoseconds(0),
      _maxNumPyramids(std::numeric_limits<int>::max()),  // Don't limit the number of generated pyramids by default
      _allocatedComputationTime(0),  // To be set when the planner is called
      _numTrajectoriesGenerated(0),
      _numCollisionChecks(0),
      _pyramidSearchPixelBuffer(2)  // A sample point must be more than 2 pixels away from the edge of a pyramid to use that pyramid for collision checking
{
  _depthData = reinterpret_cast<const uint16_t*>(depthImage.data);
}

bool DepthImagePlanner::FindFastestTrajRandomCandidates(
    RapidQuadrocopterTrajectoryGenerator::RapidTrajectoryGenerator& trajectory,
    double allocatedComputationTime, CommonMath::Vec3 explorationDirection) {

  ExplorationCost explorationCost(explorationDirection);
  return FindLowestCostTrajectoryRandomCandidates(
      trajectory, allocatedComputationTime, &explorationCost,
      &ExplorationCost::ExplorationDirectionCostWrapper);
}

bool DepthImagePlanner::FindLowestCostTrajectoryRandomCandidates(
    RapidQuadrocopterTrajectoryGenerator::RapidTrajectoryGenerator& trajectory,
    double allocatedComputationTime,
    void* costFunctionDefinitionObject,
    double (*costFunctionWrapper)(
        void* costFunctionDefinitionObject,
        RapidQuadrocopterTrajectoryGenerator::RapidTrajectoryGenerator&)) {

  RandomTrajectoryGenerator trajGenObj(this);
  return FindLowestCostTrajectory(
      trajectory, allocatedComputationTime, costFunctionDefinitionObject,
      costFunctionWrapper, (void*) &trajGenObj,
      &RandomTrajectoryGenerator::GetNextCandidateTrajectoryWrapper);
}

bool DepthImagePlanner::FindLowestCostTrajectory(
    RapidQuadrocopterTrajectoryGenerator::RapidTrajectoryGenerator& trajectory,
    double allocatedComputationTime,
    void* costFunctionDefinitionObject,
    double (*costFunctionWrapper)(
        void* costFunctionDefinitionObject,
        RapidQuadrocopterTrajectoryGenerator::RapidTrajectoryGenerator&),
    void* trajectoryGeneratorObject,
    int (*trajectoryGeneratorWrapper)(
        void* trajectoryGeneratorObject,
        RapidQuadrocopterTrajectoryGenerator::RapidTrajectoryGenerator& nextTraj)) {

  // Start timing the planner
  _startTime = high_resolution_clock::now();
  _allocatedComputationTime = allocatedComputationTime;

  bool feasibleTrajFound = false;
  double bestCost = std::numeric_limits<double>::max();

  // Get the initial state of the vehicle so we can initialize all of the candidate trajectories
  Vec3 pos0 = trajectory.GetPosition(0);
  Vec3 vel0 = trajectory.GetVelocity(0);
  Vec3 acc0 = trajectory.GetAcceleration(0);
  Vec3 grav = trajectory.GetGravityVector();

  // We assume that the initial state is written in the camera-fixed frame
  assert(pos0.x == 0 && pos0.y == 0 && pos0.z == 0);

  RapidTrajectoryGenerator candidateTraj(pos0, vel0, acc0, grav);
  while (true) {

    if (duration_cast<microseconds>(high_resolution_clock::now() - _startTime)
        .count() > int(_allocatedComputationTime * 1e6)) {
      break;
    }

    // Get the next candidate trajectory to evaluate using the provided trajectory generator
    int returnVal = (*trajectoryGeneratorWrapper)(trajectoryGeneratorObject,
                                                  candidateTraj);
    if (returnVal < 0) {
      // There are no more candidate trajectories to check. This case should only be reached if
      // the candidate trajectory generator is designed to only give a finite number of candidates
      // (e.g. when using a gridded approach instead of using random search)
      break;
    }
    _numTrajectoriesGenerated++;

    // Compute the cost of the trajectory using the provided cost function
    double cost = (*costFunctionWrapper)(costFunctionDefinitionObject,
                                         candidateTraj);
    if (cost < bestCost) {
      // The trajectory is a lower cost than lowest cost trajectory found so far

      // Check whether the trajectory is dynamically feasible
      RapidTrajectoryGenerator::InputFeasibilityResult res = candidateTraj
          .CheckInputFeasibility(_minimumAllowedThrust, _maximumAllowedThrust,
                                 _maximumAllowedAngularVelocity,
                                 _minimumSectionTimeDynamicFeas);
      if (res == RapidTrajectoryGenerator::InputFeasible) {
        // The trajectory is dynamically feasible

        // Check whether the trajectory collides with obstacles
        bool isCollisionFree = IsCollisionFree(candidateTraj.GetTrajectory());
        _numCollisionChecks++;
        if (isCollisionFree) {
          feasibleTrajFound = true;
          bestCost = cost;
          trajectory = RapidTrajectoryGenerator(candidateTraj);
        }
      }
    }
  }

  if (feasibleTrajFound) {
    return true;
  } else {
    return false;
  }
}

bool DepthImagePlanner::IsCollisionFree(Trajectory trajectory) {

  // Split trajectory into sections with monotonically changing depth
  std::vector<MonotonicTrajectory> monotonicSections = GetMonotonicSections(
      trajectory);
  while (monotonicSections.size() > 0) {

    // Check if we've used up all of our computation time
    if (duration_cast<microseconds>(high_resolution_clock::now() - _startTime)
        .count() > int(_allocatedComputationTime * 1e6)) {
      return false;
    }

    // Get a monotonic section to check
    MonotonicTrajectory monoTraj = monotonicSections.back();
    monotonicSections.pop_back();

    // Find the pixel corresponding to the endpoint of this section (deepest depth)
    Vec3 startPoint, endPoint;
    if (monoTraj.increasingDepth) {
      startPoint = monoTraj.GetValue(monoTraj.GetStartTime());
      endPoint = monoTraj.GetValue(monoTraj.GetEndTime());
    } else {
      startPoint = monoTraj.GetValue(monoTraj.GetEndTime());
      endPoint = monoTraj.GetValue(monoTraj.GetStartTime());
    }

    // Ignore the trajectory section if it's closer than the minimum collision checking distance
    if (startPoint.z < _minCheckingDist && endPoint.z < _minCheckingDist) {
      continue;
    }

    // Try to find pyramid that contains endPoint
    double endPointPixel[2];
    ProjectPointToPixel(endPoint, endPointPixel[0], endPointPixel[1]);
    Pyramid collisionCheckPyramid;
    bool pyramidFound = FindContainingPyramid(endPointPixel[0],
                                              endPointPixel[1], endPoint.z,
                                              collisionCheckPyramid);
    if (!pyramidFound) {
      // No pyramids containing endPoint were found, try to make a new pyramid
      if (_pyramids.size() >= _maxNumPyramids
          || _pyramidGenTimeNanoseconds > _maxPyramidGenTime * 1e9) {
        // We've already exceeded the maximum number of allowed pyramids or
        // the maximum time allocated for pyramid generation.
        return false;
      }

      high_resolution_clock::time_point startInflate =
          high_resolution_clock::now();
      bool pyramidGenerated = InflatePyramid(endPointPixel[0], endPointPixel[1],
                                             endPoint.z, collisionCheckPyramid);
      _pyramidGenTimeNanoseconds += duration_cast<nanoseconds>(
          high_resolution_clock::now() - startInflate).count();

      if (pyramidGenerated) {
        // Insert the new pyramid into the list of pyramids found so far
        auto index = std::lower_bound(_pyramids.begin(), _pyramids.end(),
                                      collisionCheckPyramid.depth);
        _pyramids.insert(index, collisionCheckPyramid);
      } else {
        // No pyramid could be formed, so there must be a collision
        return false;
      }
    }

    // Check if/when the trajectory intersects a lateral face of the given pyramid.
    double collisionTime;
    bool collidesWithPyramid = FindDeepestCollisionTime(monoTraj,
                                                        collisionCheckPyramid,
                                                        collisionTime);

    if (collidesWithPyramid) {
      // The trajectory collides with at least lateral face of the pyramid. Split the trajectory where it intersects,
      // and add the section outside the pyramid for further collision checking.
      if (monoTraj.increasingDepth) {
        monotonicSections.push_back(
            MonotonicTrajectory(monoTraj.GetCoeffs(), monoTraj.GetStartTime(),
                                collisionTime));
      } else {
        monotonicSections.push_back(
            MonotonicTrajectory(monoTraj.GetCoeffs(), collisionTime,
                                monoTraj.GetEndTime()));
      }
    }
  }
  return true;
}

std::vector<MonotonicTrajectory> DepthImagePlanner::GetMonotonicSections(
    Trajectory trajectory) {
  // This function exploits the property described in Section II.B of the RAPPIDS paper

  // Compute the coefficients of \dot{d}_z(t)
  std::vector<Vec3> trajDerivativeCoeffs = trajectory.GetDerivativeCoeffs();
  double c[5] = { trajDerivativeCoeffs[0].z, trajDerivativeCoeffs[1].z,
      trajDerivativeCoeffs[2].z, trajDerivativeCoeffs[3].z,
      trajDerivativeCoeffs[4].z };  // Just shortening the names

  // Compute the times at which the trajectory changes direction along the z-axis
  double roots[6];
  roots[0] = trajectory.GetStartTime();
  roots[1] = trajectory.GetEndTime();
  size_t rootCount;
  if (fabs(c[0]) > 1e-6) {
    rootCount = Quartic::solve_quartic(c[1] / c[0], c[2] / c[0], c[3] / c[0],
                                       c[4] / c[0], roots + 2);
  } else {
    rootCount = Quartic::solveP3(c[2] / c[1], c[3] / c[1], c[4] / c[1],
                                 roots + 2);
  }
  std::sort(roots, roots + rootCount + 2);  // Use rootCount + 2 because we include the start and end point

  std::vector<MonotonicTrajectory> monotonicSections;
// We don't iterate until rootCount + 2 because we need to find pairs of roots
  for (unsigned i = 0; i < rootCount + 1; i++) {
    if (roots[i] < trajectory.GetStartTime()) {
      // Skip root if it's before start time
      continue;
    } else if (fabs(roots[i] - roots[i + 1]) < 1e-6) {
      // Skip root because it's a duplicate
      continue;
    } else if (roots[i] >= trajectory.GetEndTime()) {
      // We're done because the roots are in ascending order
      break;
    }
    // Add a section between the current root and the next root after checking that the next root is valid
    // We already know that roots[i+1] is greater than the start time because roots[i] is greater than the start time and roots is sorted
    if (roots[i + 1] <= trajectory.GetEndTime()) {
      monotonicSections.push_back(
          MonotonicTrajectory(trajectory.GetCoeffs(), roots[i], roots[i + 1]));
    } else {
      // We're done because the next section is out of the range
      break;
    }
  }
  std::sort(monotonicSections.begin(), monotonicSections.end());
  return monotonicSections;
}

bool DepthImagePlanner::FindContainingPyramid(double pixelX, double pixelY,
                                              double depth,
                                              Pyramid &outPyramid) {
  // This function searches _pyramids for those with base planes at deeper
  // depths than endPoint.z
  auto firstPyramidIndex = std::lower_bound(_pyramids.begin(), _pyramids.end(),
                                            depth);
  if (firstPyramidIndex != _pyramids.end()) {
    // At least one pyramid exists that has a base plane deeper than endPoint.z
    for (std::vector<Pyramid>::iterator it = firstPyramidIndex;
        it != _pyramids.end(); ++it) {
      // Check whether endPoint is inside the pyramid
      // We need to use the _pyramidSearchPixelBuffer offset here because otherwise we'll try to
      // collision check with the pyramid we just exited while checking the previous section
      if ((*it).leftPixBound + _pyramidSearchPixelBuffer < pixelX
          && pixelX < (*it).rightPixBound - _pyramidSearchPixelBuffer
          && (*it).topPixBound + _pyramidSearchPixelBuffer < pixelY
          && pixelY < (*it).bottomPixBound - _pyramidSearchPixelBuffer) {
        outPyramid = *it;
        return true;
      }
    }
  }
  return false;
}

bool DepthImagePlanner::FindDeepestCollisionTime(MonotonicTrajectory monoTraj,
                                                 Pyramid pyramid,
                                                 double& outCollisionTime) {
  // This function exploits the property described in Section II.C of the RAPPIDS paper

  bool collidesWithPyramid = false;
  if (monoTraj.increasingDepth) {
    outCollisionTime = monoTraj.GetStartTime();
  } else {
    outCollisionTime = monoTraj.GetEndTime();
  }
  std::vector<Vec3> coeffs = monoTraj.GetCoeffs();
  for (Vec3 normal : pyramid.planeNormals) {

    // Compute the coefficients of d(t) (distance to the lateral face of the pyramid)
    double c[5] = { 0, 0, 0, 0, 0 };
    for (int dim = 0; dim < 3; dim++) {
      c[0] += normal[dim] * coeffs[0][dim];  //t**5
      c[1] += normal[dim] * coeffs[1][dim];  //t**4
      c[2] += normal[dim] * coeffs[2][dim];  //t**3
      c[3] += normal[dim] * coeffs[3][dim];  //t**2
      c[4] += normal[dim] * coeffs[4][dim];  //t
      // coeffs[5] = (0,0,0) because trajectory is at (0,0,0) at t = 0
    }

    // Find the times at which the trajectory intersects the plane
    double roots[4];
    size_t rootCount;
    if (fabs(c[0]) > 1e-6) {
      rootCount = Quartic::solve_quartic(c[1] / c[0], c[2] / c[0], c[3] / c[0],
                                         c[4] / c[0], roots);
    } else {
      rootCount = Quartic::solveP3(c[2] / c[1], c[3] / c[1], c[4] / c[1],
                                   roots);
    }
    std::sort(roots, roots + rootCount);
    if (monoTraj.increasingDepth) {
      // Search backward in time (decreasing depth)
      for (int i = rootCount - 1; i >= 0; i--) {
        if (roots[i] > monoTraj.GetEndTime()) {
          continue;
        } else if (roots[i] > monoTraj.GetStartTime()) {
          if (roots[i] > outCollisionTime) {
            // This may seem unnecessary because we are searching an ordered list, but this check is needed
            // because we are checking multiple lateral faces of the pyramid for collisions
            outCollisionTime = roots[i];
            collidesWithPyramid = true;
            break;
          }
        } else {
          break;
        }
      }
    } else {
      // Search forward in time (decreasing depth)
      for (int i = 0; i < int(rootCount); i++) {
        if (roots[i] < monoTraj.GetStartTime()) {
          continue;
        } else if (roots[i] < monoTraj.GetEndTime()) {
          if (roots[i] < outCollisionTime) {
            outCollisionTime = roots[i];
            collidesWithPyramid = true;
            break;
          }
        } else {
          break;
        }
      }
    }
  }
  return collidesWithPyramid;
}

bool DepthImagePlanner::InflatePyramid(int x0, int y0, double minimumDepth,
                                       Pyramid &outPyramid) {
  // This function is briefly described by Section III.A. of the RAPPIDS paper

  // First check if the sample point violates the field of view constraints
  int imageEdgeOffset = _focalLength * _trueVehicleRadius / _minCheckingDist;
  if (x0 <= imageEdgeOffset + _pyramidSearchPixelBuffer + 1
      || x0 > _imageWidth - imageEdgeOffset - _pyramidSearchPixelBuffer - 1
      || y0 <= imageEdgeOffset + _pyramidSearchPixelBuffer + 1
      || y0 > _imageHeight - imageEdgeOffset - _pyramidSearchPixelBuffer - 1) {
    // Sample point could be in collision with something outside the FOV
    return false;
  }

  // The base plane of the pyramid must be deeper than this depth (written in pixel depth units)
  uint16_t minimumPyramidDepth = uint16_t(
      (minimumDepth + _vehicleRadiusForPlanning) / _depthScale);

  // This is the minimum "radius" (really the width/height divided by two) of a valid pyramid
  int initPixSearchRadius = _focalLength * _vehicleRadiusForPlanning
      / (_depthScale * minimumPyramidDepth);

  if (2 * initPixSearchRadius
      >= std::min(_imageWidth, _imageHeight) - 2 * imageEdgeOffset) {
    // The minimum size of the pyramid is larger than the maximum pyramid size
    return false;
  }

  // These edges are the edges of the expanded pyramid before it is shrunk to the final size
  int leftEdge, topEdge, rightEdge, bottomEdge;
  if (y0 - initPixSearchRadius < imageEdgeOffset) {
    topEdge = imageEdgeOffset;
    bottomEdge = topEdge + 2 * initPixSearchRadius;
  } else {
    bottomEdge = std::min(_imageHeight - imageEdgeOffset - 1,
                          y0 + initPixSearchRadius);
    topEdge = bottomEdge - 2 * initPixSearchRadius;
  }
  if (x0 - initPixSearchRadius < imageEdgeOffset) {
    leftEdge = imageEdgeOffset;
    rightEdge = leftEdge + 2 * initPixSearchRadius;
  } else {
    rightEdge = std::min(_imageWidth - imageEdgeOffset - 1,
                         x0 + initPixSearchRadius);
    leftEdge = rightEdge - 2 * initPixSearchRadius;
  }

  // We don't look at any pixels closer than this distance (e.g. if the propellers are in the field of view)
  uint16_t ignoreDist = uint16_t(_trueVehicleRadius / _depthScale);
  // For reading the depth value stored in a given pixel.
  uint16_t pixDist;

  for (int y = topEdge; y < bottomEdge; y++) {
    for (int x = leftEdge; x < rightEdge; x++) {
      pixDist = _depthData[y * _imageWidth + x];
      if (pixDist <= minimumPyramidDepth && pixDist > ignoreDist) {
        // We are unable to inflate a rectangle that will meet the minimum size requirements
        return false;
      }
    }
  }

  // Store the minimum depth pixel value of the expanded pyramid. The base plane of the final pyramid will be
  // this value minus the vehicle radius.
  uint16_t maxDepthExpandedPyramid = std::numeric_limits<uint16_t>::max();

  // We search each edge of the rectangle until we hit a pixel value closer than minimumPyramidDepth
  // This creates a spiral search pattern around the initial sample point
  // Once all four sides of the pyramid hit either the FOV constraint or a pixel closer than
  // minimumPyramidDepth, we will shrink the pyramid based on the vehicle radius.
  bool rightFree = true, topFree = true, leftFree = true, bottomFree = true;
  while (rightFree || topFree || leftFree || bottomFree) {
    if (rightFree) {
      if (rightEdge < _imageWidth - imageEdgeOffset - 1) {
        for (int y = topEdge; y <= bottomEdge; y++) {
          pixDist = _depthData[y * _imageWidth + rightEdge + 1];
          if (pixDist > ignoreDist) {
            if (pixDist < minimumPyramidDepth) {
              rightFree = false;
              rightEdge--;  // Negate the ++ after breaking loop
              break;
            }
            maxDepthExpandedPyramid = std::min(maxDepthExpandedPyramid,
                                               pixDist);
          }
        }
        rightEdge++;
      } else {
        rightFree = false;
      }
    }
    if (topFree) {
      if (topEdge > imageEdgeOffset) {
        for (int x = leftEdge; x <= rightEdge; x++) {
          pixDist = _depthData[(topEdge - 1) * _imageWidth + x];
          if (pixDist > ignoreDist) {
            if (pixDist < minimumPyramidDepth) {
              topFree = false;
              topEdge++;  // Negate the -- after breaking loop
              break;
            }
            maxDepthExpandedPyramid = std::min(maxDepthExpandedPyramid,
                                               pixDist);
          }
        }
        topEdge--;
      } else {
        topFree = false;
      }
    }
    if (leftFree) {
      if (leftEdge > imageEdgeOffset) {
        for (int y = topEdge; y <= bottomEdge; y++) {
          pixDist = _depthData[y * _imageWidth + leftEdge - 1];
          if (pixDist > ignoreDist) {
            if (pixDist < minimumPyramidDepth) {
              leftFree = false;
              leftEdge++;  // Negate the -- after breaking loop
              break;
            }
            maxDepthExpandedPyramid = std::min(maxDepthExpandedPyramid,
                                               pixDist);
          }
        }
        leftEdge--;
      } else {
        leftFree = false;
      }
    }
    if (bottomFree) {
      if (bottomEdge < _imageHeight - imageEdgeOffset - 1) {
        for (int x = leftEdge; x <= rightEdge; x++) {
          pixDist = _depthData[(bottomEdge + 1) * _imageWidth + x];
          if (pixDist > ignoreDist) {
            if (pixDist < minimumPyramidDepth) {
              bottomFree = false;
              bottomEdge--;  // Negate the ++ after breaking loop
              break;
            }
            maxDepthExpandedPyramid = std::min(maxDepthExpandedPyramid,
                                               pixDist);
          }
        }
        bottomEdge++;
      } else {
        bottomFree = false;
      }
    }
  }

  // Next, shrink the pyramid according to the vehicle radius
  // Number of pixels to shrink final pyramid. Found by searching outside the boundaries of the expanded pyramid.
  // These edges will be the edges of the final pyramid.
  int rightEdgeShrunk = _imageWidth - 1 - imageEdgeOffset;
  int leftEdgeShrunk = imageEdgeOffset;
  int topEdgeShrunk = imageEdgeOffset;
  int bottomEdgeShrunk = _imageHeight - 1 - imageEdgeOffset;
  int numerator = _focalLength * _vehicleRadiusForPlanning / _depthScale;

  // First check the area between each edge and the edge of the image
  // Check right side
  for (int x = rightEdge; x < _imageWidth; x++) {
    for (int y = topEdge; y <= bottomEdge; y++) {
      pixDist = _depthData[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        // The pixel is farther away than the minimum checking distance
        if (numerator > (x - rightEdgeShrunk) * pixDist) {
          int rightShrinkTemp = x - int(numerator / pixDist);
          if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking from right will make pyramid invalid
            // Can we shrink from top or bottom instead?
            int topShrinkTemp = y + int(numerator / pixDist);
            int bottomShrinkTemp = y - int(numerator / pixDist);
            if (y0 < topShrinkTemp + _pyramidSearchPixelBuffer
                && y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
              // We can't shrink either edge
              return false;
            } else if (y0 < topShrinkTemp + _pyramidSearchPixelBuffer) {
              // We can't shrink the upper edge, so shrink the lower edge
              bottomEdgeShrunk = bottomShrinkTemp;
            } else if (y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
              // We can't shrink the lower edge, so shrink the upper edge
              topEdgeShrunk = topShrinkTemp;
            } else {
              // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
              int uShrinkLostArea = (topShrinkTemp - topEdgeShrunk);
              int dShrinkLostArea = (bottomEdgeShrunk - bottomShrinkTemp);
              if (dShrinkLostArea > uShrinkLostArea) {
                // We lose more area shrinking the bottom side, so shrink the top side
                topEdgeShrunk = topShrinkTemp;
              } else {
                // We lose more area shrinking the top side, so shrink the bottom side
                rightEdgeShrunk = bottomShrinkTemp;
              }
            }
          } else {
            rightEdgeShrunk = rightShrinkTemp;
          }
        }
      }
    }
  }
  // Check left side
  for (int x = leftEdge; x >= 0; x--) {
    for (int y = topEdge; y <= bottomEdge; y++) {
      pixDist = _depthData[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        if ((leftEdgeShrunk - x) * pixDist < numerator) {
          int leftShrinkTemp = x + int(numerator / pixDist);
          if (x0 < leftShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking from left will make pyramid invalid
            // Can we shrink from top or bottom instead?
            int topShrinkTemp = y + int(numerator / pixDist);
            int bottomShrinkTemp = y - int(numerator / pixDist);
            if (y0 < topShrinkTemp + _pyramidSearchPixelBuffer
                && y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
              // We can't shrink either edge
              return false;
            } else if (y0 < topShrinkTemp + _pyramidSearchPixelBuffer) {
              // We can't shrink the upper edge, so shrink the lower edge
              bottomEdgeShrunk = bottomShrinkTemp;
            } else if (y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
              // We can't shrink the lower edge, so shrink the upper edge
              topEdgeShrunk = topShrinkTemp;
            } else {
              // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
              int uShrinkLostArea = (topShrinkTemp - topEdgeShrunk);
              int dShrinkLostArea = (bottomEdgeShrunk - bottomShrinkTemp);
              if (dShrinkLostArea > uShrinkLostArea) {
                // We lose more area shrinking the bottom side, so shrink the top side
                topEdgeShrunk = topShrinkTemp;
              } else {
                // We lose more area shrinking the top side, so shrink the bottom side
                bottomEdgeShrunk = bottomShrinkTemp;
              }
            }
          } else {
            leftEdgeShrunk = leftShrinkTemp;
          }
        }
      }
    }
  }
  if (leftEdgeShrunk + _pyramidSearchPixelBuffer
      > rightEdgeShrunk - _pyramidSearchPixelBuffer) {
    // We shrunk the left and right sides so much that the pyramid is too small!
    return false;
  }

  // Check top side
  for (int y = topEdge; y >= 0; y--) {
    for (int x = leftEdge; x <= rightEdge; x++) {
      pixDist = _depthData[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        if ((topEdgeShrunk - y) * pixDist < numerator) {
          int topShrinkTemp = y + int(numerator / pixDist);
          if (y0 < topShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking from top will make pyramid invalid
            // Can we shrink from left or right instead?
            int rightShrinkTemp = x - int(numerator / pixDist);
            int leftShrinkTemp = x + int(numerator / pixDist);
            if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer
                && x0 < leftShrinkTemp + _pyramidSearchPixelBuffer) {
              // We can't shrink either edge
              return false;
            } else if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer) {
              // We can't shrink the upper right, so shrink the left edge
              leftEdgeShrunk = leftShrinkTemp;
            } else if (x0 < leftShrinkTemp + _pyramidSearchPixelBuffer) {
              // We can't shrink the left edge, so shrink the right edge
              rightEdgeShrunk = rightShrinkTemp;
            } else {
              // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
              int rShrinkLostArea = (rightEdgeShrunk - rightShrinkTemp);
              int lShrinkLostArea = (leftShrinkTemp - leftEdgeShrunk);
              if (rShrinkLostArea > lShrinkLostArea) {
                // We lose more area shrinking the right side, so shrink the left side
                leftEdgeShrunk = leftShrinkTemp;
              } else {
                // We lose more area shrinking the left side, so shrink the right side
                rightEdgeShrunk = rightShrinkTemp;
              }
            }
          } else {
            topEdgeShrunk = topShrinkTemp;
          }
        }
      }
    }
  }
  // Check bottom side
  for (int y = bottomEdge; y < _imageHeight; y++) {
    for (int x = leftEdge; x <= rightEdge; x++) {
      pixDist = _depthData[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        // The pixel is farther away than the minimum checking distance
        if (numerator > (y - bottomEdgeShrunk) * pixDist) {
          int bottomShrinkTemp = y - int(numerator / pixDist);
          if (y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking from top will make pyramid invalid
            // Can we shrink from left or right instead?
            int rightShrinkTemp = x - int(numerator / pixDist);
            int leftShrinkTemp = x + int(numerator / pixDist);
            if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer
                && x0 < leftShrinkTemp + _pyramidSearchPixelBuffer) {
              // We can't shrink either edge
              return false;
            } else if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer) {
              // We can't shrink the upper right, so shrink the left edge
              leftEdgeShrunk = leftShrinkTemp;
            } else if (x0 < leftShrinkTemp + _pyramidSearchPixelBuffer) {
              // We can't shrink the left edge, so shrink the right edge
              rightEdgeShrunk = rightShrinkTemp;
            } else {
              // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
              int rShrinkLostArea = (rightEdgeShrunk - rightShrinkTemp);
              int lShrinkLostArea = (leftShrinkTemp - leftEdgeShrunk);
              if (rShrinkLostArea > lShrinkLostArea) {
                // We lose more area shrinking the right side, so shrink the left side
                leftEdgeShrunk = leftShrinkTemp;
              } else {
                // We lose more area shrinking the left side, so shrink the right side
                rightEdgeShrunk = rightShrinkTemp;
              }
            }
          } else {
            bottomEdgeShrunk = bottomShrinkTemp;
          }
        }
      }
    }
  }
  if (topEdgeShrunk + _pyramidSearchPixelBuffer
      > bottomEdgeShrunk - _pyramidSearchPixelBuffer) {
    // We shrunk the top and bottom sides so much that the pyramid has no volume!
    return false;
  }

  // Next, check the corners that we ignored before
  // Check top right corner
  for (int y = topEdge; y >= 0; y--) {
    for (int x = rightEdge; x < _imageWidth; x++) {
      pixDist = _depthData[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        if (numerator > (x - rightEdgeShrunk) * pixDist
            && (topEdgeShrunk - y) * pixDist < numerator) {
          // Both right and top edges could shrink
          int rightShrinkTemp = x - int(numerator / pixDist);
          int topShrinkTemp = y + int(numerator / pixDist);
          if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer
              && y0 < topShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking either edge makes the pyramid exclude the starting point
            return false;
          } else if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking right edge makes pyramid exclude the starting point, so shrink the top edge
            topEdgeShrunk = topShrinkTemp;
          } else if (y0 < topShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking top edge makes pyramid exclude the starting point, so shrink the right edge
            rightEdgeShrunk = rightShrinkTemp;
          } else {
            // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
            int rShrinkLostArea = (rightEdgeShrunk - rightShrinkTemp)
                * (bottomEdgeShrunk - topEdgeShrunk);
            int uShrinkLostArea = (topShrinkTemp - topEdgeShrunk)
                * (rightEdgeShrunk - leftEdgeShrunk);
            if (rShrinkLostArea > uShrinkLostArea) {
              // We lose more area shrinking the right side, so shrink the top side
              topEdgeShrunk = topShrinkTemp;
            } else {
              // We lose more area shrinking the top side, so shrink the right side
              rightEdgeShrunk = rightShrinkTemp;
            }
          }
        }
      }
    }
  }
  // Check bottom right corner
  for (int y = bottomEdge; y < _imageHeight; y++) {
    for (int x = rightEdge; x < _imageWidth; x++) {
      pixDist = _depthData[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        if (numerator > (x - rightEdgeShrunk) * pixDist
            && numerator > (y - bottomEdgeShrunk) * pixDist) {
          // Both right and bottom edges could shrink
          int rightShrinkTemp = x - int(numerator / pixDist);
          int bottomShrinkTemp = y - int(numerator / pixDist);
          if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer
              && y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking either edge makes the pyramid exclude the starting point
            return false;
          } else if (x0 > rightShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking right edge makes pyramid exclude the starting point, so shrink the bottom edge
            bottomEdgeShrunk = bottomShrinkTemp;
          } else if (y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking bottom edge makes pyramid exclude the starting point, so shrink the right edge
            rightEdgeShrunk = rightShrinkTemp;
          } else {
            // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
            int rShrinkLostArea = (rightEdgeShrunk - rightShrinkTemp)
                * (bottomEdgeShrunk - topEdgeShrunk);
            int dShrinkLostArea = (bottomEdgeShrunk - bottomShrinkTemp)
                * (rightEdgeShrunk - leftEdgeShrunk);
            if (rShrinkLostArea > dShrinkLostArea) {
              // We lose more area shrinking the right side, so shrink the bottom side
              bottomEdgeShrunk = bottomShrinkTemp;
            } else {
              // We lose more area shrinking the bottom side, so shrink the right side
              rightEdgeShrunk = rightShrinkTemp;
            }
          }
        }
      }
    }
  }
  // Check top left corner
  for (int y = topEdge; y >= 0; y--) {
    for (int x = leftEdge; x >= 0; x--) {
      pixDist = _depthData[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        if ((leftEdgeShrunk - x) * pixDist < numerator
            && (topEdgeShrunk - y) * pixDist < numerator) {
          // Both left and top edges could shrink
          int leftShrinkTemp = x + int(numerator / pixDist);
          int topShrinkTemp = y + int(numerator / pixDist);
          if (x0 < leftShrinkTemp + _pyramidSearchPixelBuffer
              && y0 < topShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking either edge makes the pyramid exclude the starting point
            return false;
          } else if (x0 < leftShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking left edge makes pyramid exclude the starting point, so shrink the top edge
            topEdgeShrunk = topShrinkTemp;
          } else if (y0 < topShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking top edge makes pyramid exclude the starting point, so shrink the left edge
            leftEdgeShrunk = leftShrinkTemp;
          } else {
            // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
            int lShrinkLostArea = (leftShrinkTemp - leftEdgeShrunk)
                * (bottomEdgeShrunk - topEdgeShrunk);
            int uShrinkLostArea = (topShrinkTemp - topEdgeShrunk)
                * (rightEdgeShrunk - leftEdgeShrunk);
            if (lShrinkLostArea > uShrinkLostArea) {
              // We lose more area shrinking the left side, so shrink the top side
              topEdgeShrunk = topShrinkTemp;
            } else {
              // We lose more area shrinking the top side, so shrink the left side
              leftEdgeShrunk = leftShrinkTemp;
            }
          }
        }
      }
    }
  }
  // Check bottom left corner
  for (int y = bottomEdge; y < _imageHeight; y++) {
    for (int x = leftEdge; x >= 0; x--) {
      pixDist = _depthData[y * _imageWidth + x];
      if (pixDist > ignoreDist && pixDist < maxDepthExpandedPyramid) {
        if ((leftEdgeShrunk - x) * pixDist < numerator
            && numerator > (y - bottomEdgeShrunk) * pixDist) {
          // Both left and bottom edges could shrink
          int leftShrinkTemp = x + int(numerator / pixDist);
          int bottomShrinkTemp = y - int(numerator / pixDist);
          if (x0 < leftShrinkTemp + _pyramidSearchPixelBuffer
              && y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking either edge makes the pyramid exclude the starting point
            return false;
          } else if (x0 < leftShrinkTemp + _pyramidSearchPixelBuffer) {
            // Shrinking left edge makes pyramid exclude the starting point, so shrink the bottom edge
            bottomEdgeShrunk = bottomShrinkTemp;
          } else if (y0 > bottomShrinkTemp - _pyramidSearchPixelBuffer) {
            // Shrinking bottom edge makes pyramid exclude the starting point, so shrink the left edge
            leftEdgeShrunk = leftShrinkTemp;
          } else {
            // We can shrink either edge and still have a feasible pyramid, choose the edge that removes the least area
            int lShrinkLostArea = (leftShrinkTemp - leftEdgeShrunk)
                * (bottomEdgeShrunk - topEdgeShrunk);
            int dShrinkLostArea = (bottomEdgeShrunk - bottomShrinkTemp)
                * (rightEdgeShrunk - leftEdgeShrunk);
            if (lShrinkLostArea > dShrinkLostArea) {
              // We lose more area shrinking the left side, so shrink the bottom side
              bottomEdgeShrunk = bottomShrinkTemp;
            } else {
              // We lose more area shrinking the bottom side, so shrink the left side
              leftEdgeShrunk = leftShrinkTemp;
            }
          }
        }
      }
    }
  }

  int edgesFinal[4] = { rightEdgeShrunk, topEdgeShrunk, leftEdgeShrunk,
      bottomEdgeShrunk };
  double depth = maxDepthExpandedPyramid * _depthScale
      - _vehicleRadiusForPlanning;

  // Create a new pyramid
  Vec3 corners[4];
  // Top right
  DeprojectPixelToPoint(double(edgesFinal[0]), double(edgesFinal[1]), depth,
                        corners[0]);
  // Top left
  DeprojectPixelToPoint(double(edgesFinal[2]), double(edgesFinal[1]), depth,
                        corners[1]);
  // Bottom left
  DeprojectPixelToPoint(double(edgesFinal[2]), double(edgesFinal[3]), depth,
                        corners[2]);
  // Bottom right
  DeprojectPixelToPoint(double(edgesFinal[0]), double(edgesFinal[3]), depth,
                        corners[3]);
  outPyramid = Pyramid(depth, edgesFinal, corners);

  return true;
}

void DepthImagePlanner::MeasureConservativeness(
    int numTrajToEvaluate, int pyramidLimit,
    RapidQuadrocopterTrajectoryGenerator::RapidTrajectoryGenerator& trajectory,
    int &numIncorrectInCollision, int &numCorrectInCollision) {
  // The test described in Section IV.A of the RAPPIDS paper

  SetMaxNumberOfPyramids(pyramidLimit);
  numIncorrectInCollision = 0;
  numCorrectInCollision = 0;

  // Generate trajectories used to evaluate the collision checker
  RandomTrajectoryGenerator trajGenObj(this);
  std::vector<Trajectory> trajs;
  trajs.reserve(numTrajToEvaluate);
  for (int i = 0; i < numTrajToEvaluate; i++) {
    trajGenObj.GetNextCandidateTrajectory(trajectory);
    trajs.push_back(trajectory.GetTrajectory());
  }

  // Run our collision checker and compare to the results of the ground truth ray-tracing based method
  for (auto candidateTraj : trajs) {
    _startTime = std::chrono::high_resolution_clock::now();  // Prevents collision checking from ending early due to timing constraints
    _allocatedComputationTime = 1000;
    bool collidesPlanner = !IsCollisionFree(candidateTraj);
    bool collidesGroundTruth = !IsCollisionFreeGroundTruth(candidateTraj);
    if (collidesGroundTruth && collidesPlanner) {
      numCorrectInCollision++;
    } else if (collidesPlanner && !collidesGroundTruth) {
      numIncorrectInCollision++;
    }
  }
}

void DepthImagePlanner::MeasureCollisionCheckingSpeed(
    int numTrajToEvaluate, double pyramidGenTimeLimit,
    RapidQuadrocopterTrajectoryGenerator::RapidTrajectoryGenerator& trajectory,
    double &outTotalCollCheckTimeNs, double &outPercentCollisionFree) {
  // The test described in Section IV.B of the RAPPIDS paper

  SetMaxPyramidGenTime(pyramidGenTimeLimit);

  // Generate the trajectories used in the test
  RandomTrajectoryGenerator trajGenObj(this);
  std::vector<Trajectory> trajs;
  trajs.reserve(numTrajToEvaluate);
  for (int i = 0; i < numTrajToEvaluate; i++) {
    trajGenObj.GetNextCandidateTrajectory(trajectory);
    trajs.push_back(trajectory.GetTrajectory());
  }

  int numCollisionFree = 0;
  bool hasFeasTraj = false;

  outTotalCollCheckTimeNs = 0;
  for (auto candidateTraj : trajs) {
    _startTime = std::chrono::high_resolution_clock::now();  // Prevents collision checking from ending early due to timing constraints
    _allocatedComputationTime = 1000;
    bool isCollisionFree = IsCollisionFree(candidateTraj);
    outTotalCollCheckTimeNs += duration_cast<nanoseconds>(
        high_resolution_clock::now() - _startTime).count();
    if (isCollisionFree) {
      numCollisionFree++;
      hasFeasTraj = true;
    }
  }
  outTotalCollCheckTimeNs -= _pyramidGenTimeNanoseconds;  // Don't count time spent generating pyramids
  outPercentCollisionFree = double(numCollisionFree)
      / double(numTrajToEvaluate);

  if (hasFeasTraj) {
    outPercentCollisionFree = 1;
  }
}

bool DepthImagePlanner::IsCollisionFreeGroundTruth(Trajectory trajectory) {

  // Use this timestep to discretize the trajectory. This number should be small (but too small = very long collision checking times)
  double timestep = 0.1;
  // Ignore any pixels within the sphere around the vehicle (e.g. propellers that might be in the FOV of the depth camera)
  uint16_t ignoreDist = uint16_t(_trueVehicleRadius / _depthScale);
  // Declare any point that is too close to the edge of the image as violating the field of view constraints
  int imageEdgeOffset = _focalLength * _trueVehicleRadius / _minCheckingDist;

  // Check FOV constraints first because this is the fastest thing to check
  for (double t = trajectory.GetStartTime(); t < trajectory.GetEndTime(); t +=
      timestep) {
    Vec3 trajPos = trajectory.GetValue(t);

    if (trajPos.z < _minCheckingDist) {
      continue;
    }

    // Check FOV constraints
    double pixX, pixY;
    ProjectPointToPixel(trajPos, pixX, pixY);
    if (pixX <= imageEdgeOffset || pixX > _imageWidth - imageEdgeOffset
        || pixY <= imageEdgeOffset || pixY > _imageHeight - imageEdgeOffset) {
      // Trajectory violates FOV constraints
      return false;
    }
  }

  // For each point along the trajectory, check that each pixel doesn't occlude or intersect with the sphere around the vehicle
  for (double t = trajectory.GetStartTime(); t < trajectory.GetEndTime(); t +=
      timestep) {
    Vec3 trajPos = trajectory.GetValue(t);

    if (trajPos.z < _minCheckingDist) {
      continue;
    }

    for (int y = 0; y < _imageHeight; y++) {
      for (int x = 0; x < _imageWidth; x++) {
        // Ignore pixels that are too close
        if (_depthData[y * _imageWidth + x] > ignoreDist) {
          // Compute where the ray that pierces the current pixel intersects with the sphere around the vehicle
          Vec3 e = Vec3((x - _cx) / _focalLength, (y - _cy) / _focalLength, 1.0)
              .GetUnitVector();
          double underSqrt = pow(trajPos.Dot(e), 2) - trajPos.GetNorm2Squared()
              + pow(_vehicleRadiusForPlanning, 2);
          if (underSqrt >= 0) {
            // Ray collides with sphere of vehicle
            // Check if pixel is behind sphere
            double secondCollisionDist = e.Dot(trajPos) + sqrt(underSqrt);
            Vec3 pixPos;
            DeprojectPixelToPoint(x, y,
                                  _depthData[y * _imageWidth + x] * _depthScale,
                                  pixPos);
            double pixDist = pixPos.GetNorm2();
            if (pixDist < secondCollisionDist) {
              // Pixel is inside or in front of the vehicle sphere
              return false;
            }
          }
          // Else: ray does not collide with vehicle sphere
        }
      }
    }
  }

  return true;
}

