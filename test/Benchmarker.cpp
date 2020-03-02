/*!
 * Rectangular Pyramid Partitioning using Integrated Depth Sensors
 *
 * Copyright 2020 by Junseok Lee <junseok_lee@berkeley.edu>
 * and Nathan Bucki <nathan_bucki@berkeley.edu>
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

#include <unistd.h>
#include <boost/program_options.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <ctime>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include "RectangularPyramidPlanner/DepthImagePlanner.hpp"
#include "RapidQuadcopterTrajectories/RapidTrajectoryGenerator.hpp"

using namespace CommonMath;
using namespace RapidQuadrocopterTrajectoryGenerator;
using namespace RectangularPyramidPlanner;

enum testType {
  CONSERVATIVENESS = 0,
  COLLISION_CHECKING_TIME = 1,
  TRAJECTORY_COVERAGE = 2,
};

typedef struct {
  int w;
  int h;
  double depth_scale;
  double f;
  double cx;
  double cy;
} camera_intrinsics_t;

typedef struct {
  double physical_vehicle_radius;
  double vehicle_radius_for_planning;
  double minimum_collision_distance;
} planner_specs_t;

typedef struct {
  uint num_scenes;             // [1]
  uint num_obstacles;          // [1]
  double obstacles_width_min;  // [m]
  double obstacles_width_max;  // [m]
  double obstacles_depth_min;  // [m]
  double obstacles_depth_max;  // [m]
  uint random_seed;
} synthesis_specs_t;

typedef struct {
  bool display;
  double display_range_min;  // [m]
  double display_range_max;  // [m]
  bool png_output;
  double vx_min;
  double vx_max;
  double vy_min;
  double vy_max;
  double vz_min;
  double vz_max;
  double ay_min;
  double ay_max;
  uint random_seed;
  int test_type;
} benchmark_options_t;

typedef struct {
  std::vector<int> max_num_pyramids;
  int num_traj;
} conservative_test_options_t;

typedef struct {
  double pyramid_gen_time;
  int num_traj;
} collision_time_options_t;

typedef struct {
  double min_comp_time;
  double max_comp_time;
  int num_comp_times;
} traj_coverage_options_t;

typedef struct {
  int count;
  double min;
  double max;
  double median;
  double mean;
} benchmark_stats_t;

template<class T>
static benchmark_stats_t calculate_stats(std::vector<T>& q) {
  benchmark_stats_t stats;
  const auto median_it = q.begin() + q.size() / 2;
  std::nth_element(q.begin(), median_it, q.end());  // nth_element is a partial sorting algorithm that rearranges elements.

  stats.count = q.size();
  stats.min = *std::min_element(q.begin(), q.end());
  stats.max = *std::max_element(q.begin(), q.end());
  stats.mean = std::accumulate(std::begin(q), std::end(q), 0.0) / q.size();  // ! 0.0 double 0.0f float 0 integer
  stats.median = *median_it;

  return stats;
}

static void print_stats(std::ostream& os, const std::string& stats_name,
                        benchmark_stats_t& stats) {
  os << std::left << std::endl << std::setw(60) << "Samples:" << stats.count
     << std::endl << std::setw(60) << stats_name + ".min: " << stats.min
     << std::endl << std::setw(60) << stats_name + ".max: " << stats.max
     << std::endl << std::setw(60) << stats_name + ".mean: " << stats.mean
     << std::endl << std::setw(60) << stats_name + ".median: " << stats.median
     << std::endl;
}

static void display_image(const cv::Mat& depth_image, double pixelMin,
                          double pixelMax) {
  cv::Mat displayable_image;
  double alpha = 255.0 / (pixelMax - pixelMin);
  double beta = -alpha * pixelMin;
  depth_image.convertTo(displayable_image, CV_8UC1, alpha, beta);
  // BLUE = FAR
  // RED = CLOSE
  cv::applyColorMap(displayable_image, displayable_image, cv::COLORMAP_HOT);
  cv::imshow("display", displayable_image);
  cv::waitKey(1);
}

// Include center point of your rectangle, size of your rectangle and the degrees of rotation
static void draw_rotated_rectangle(cv::Mat& image, const uint obstacle_depth,
                                   const cv::Point centerPoint,
                                   const cv::Size rectangleSize,
                                   const double rotationDegrees) {
  cv::Scalar color = cv::Scalar(obstacle_depth);

  // Create the rotated rectangle
  cv::RotatedRect rotatedRectangle(centerPoint, rectangleSize, rotationDegrees);

  // We take the edges that OpenCV calculated for us
  cv::Point2f vertices2f[4];
  rotatedRectangle.points(vertices2f);

  // Convert them so we can use them in a fillConvexPoly
  cv::Point vertices[4];
  for (int i = 0; i < 4; ++i) {
    vertices[i] = vertices2f[i];
  }

  // Now we can fill the rotated rectangle with our specified color
  cv::fillConvexPoly(image, vertices, 4, color);
}

static void make_image(const camera_intrinsics_t& camera_intrinsics,
                       const synthesis_specs_t& synthesis_specs,
                       std::mt19937& gen, cv::Mat& out_depth_frame) {  // output
  // * Produces random floating-point values i, uniformly distributed on the interval [a, b).
  std::uniform_real_distribution<> random_angle(0, 180);
  std::uniform_real_distribution<> random_w_SI(
      synthesis_specs.obstacles_width_min, synthesis_specs.obstacles_width_max);  // [m]
  std::uniform_real_distribution<> random_depth_SI(
      synthesis_specs.obstacles_depth_min, synthesis_specs.obstacles_depth_max);  // [m]

  // * Produces random integer values i, uniformly distributed on the closed interval [a, b].
  std::uniform_int_distribution<> random_x_px(0, camera_intrinsics.w - 1);

  const uint RECT_HEIGHT = 0xFFFF;
  const uint DEPTH_MAX = 0xFFFF;  // represent infinity in 16 bit image

  cv::Mat depth_mat(camera_intrinsics.h, camera_intrinsics.w, CV_16UC1,
                    cv::Scalar(DEPTH_MAX));
  for (int j = 0; j < synthesis_specs.num_obstacles; j++) {
    int center_x = random_x_px(gen);
    int center_y = static_cast<float>(camera_intrinsics.h)
        / static_cast<float>(camera_intrinsics.w) * center_x;

    // pixelValues = depth / depth_scale
    // e.g. 1 meter = 1000 pixel value for depth_scale = 0.001
    double depth_SI = random_depth_SI(gen);
    int depth_px = depth_SI / camera_intrinsics.depth_scale;  // double to int implicit conversion
    draw_rotated_rectangle(
        depth_mat,
        depth_px,
        cv::Point2f(center_x, center_y),
        cv::Size2f(RECT_HEIGHT,
                   camera_intrinsics.f * random_w_SI(gen) / depth_SI),
        random_angle(gen));
  }

  // Output: depth frame
  out_depth_frame = depth_mat;
}

static void run_conservativeness_benchmark(
    const camera_intrinsics_t& camera_intrinsics,
    const planner_specs_t& planner_specs,
    const synthesis_specs_t& synthesis_specs,
    const benchmark_options_t& benchmark_options,
    const conservative_test_options_t& conservative_test_options,
    boost::property_tree::ptree& root) {

  if (benchmark_options.display) {
    cv::namedWindow("display", cv::WINDOW_AUTOSIZE);
  }

  // Set up the random number generators
  std::mt19937 image_rdgen(synthesis_specs.random_seed);
  std::mt19937 state_rdgen(benchmark_options.random_seed);
  std::uniform_real_distribution<> random_vx(benchmark_options.vx_min,
                                             benchmark_options.vx_max);
  std::uniform_real_distribution<> random_vy(benchmark_options.vy_min,
                                             benchmark_options.vy_max);
  std::uniform_real_distribution<> random_vz(benchmark_options.vz_min,
                                             benchmark_options.vz_max);
  std::uniform_real_distribution<> random_ay(benchmark_options.ay_min,
                                             benchmark_options.ay_max);

  // For saving the results
  boost::property_tree::ptree max_num_pyramids_ptree;

  // Run the test for each given pyramid limit
  for (int max_allowed_pyramids : conservative_test_options.max_num_pyramids) {
    int total_incorrect_in_collision = 0;
    int total_correct_in_collision = 0;
    std::vector<int> num_pyramids_generated;
    boost::property_tree::ptree conservativenesss_ptree;
    for (int i = 0; i < synthesis_specs.num_scenes; i++) {
      if (i % 20 == 0) {
        printf("Processed %d frames so far.\n", i);
      }

      // Set up the test
      cv::Mat depth_mat;
      make_image(camera_intrinsics, synthesis_specs, image_rdgen, depth_mat);
      DepthImagePlanner planner(depth_mat, camera_intrinsics.depth_scale,
                                camera_intrinsics.f, camera_intrinsics.cx,
                                camera_intrinsics.cy,
                                planner_specs.physical_vehicle_radius,
                                planner_specs.vehicle_radius_for_planning,
                                planner_specs.minimum_collision_distance);
      RapidTrajectoryGenerator traj(
          Vec3(0, 0, 0),
          Vec3(random_vx(state_rdgen), random_vy(state_rdgen),
               random_vz(state_rdgen)),
          Vec3(0, -random_ay(state_rdgen), 0), Vec3(0, 9.81, 0));

      // Run the test
      int incorrect_in_collision, correct_in_collision;
      planner.MeasureConservativeness(conservative_test_options.num_traj,
                                      max_allowed_pyramids, traj,
                                      incorrect_in_collision,
                                      correct_in_collision);

      // Store the results
      total_incorrect_in_collision += incorrect_in_collision;
      total_correct_in_collision += correct_in_collision;
      num_pyramids_generated.push_back(planner.GetPyramids().size());

      boost::property_tree::ptree convervativeness_node;
      convervativeness_node.put_value(
          double(incorrect_in_collision)
              / double(incorrect_in_collision + correct_in_collision));
      conservativenesss_ptree.push_back(
          std::make_pair("", convervativeness_node));

      if (benchmark_options.png_output) {
        std::string filename = std::to_string(i) + ".png";
        imwrite(filename, depth_mat);
      }
      if (benchmark_options.display) {
        display_image(
            depth_mat,
            benchmark_options.display_range_min / camera_intrinsics.depth_scale,
            benchmark_options.display_range_max
                / camera_intrinsics.depth_scale);
      }
    }

    benchmark_stats_t stats_pyramids = calculate_stats<int>(
        num_pyramids_generated);

    double conservativeness = double(total_incorrect_in_collision)
        / (total_incorrect_in_collision + total_correct_in_collision);

    printf("\nConservativeness = %f (%d max pyramids)\n", conservativeness,
           max_allowed_pyramids);
    print_stats(std::cout, "AvgPyramids", stats_pyramids);

    boost::property_tree::ptree maxPyramidNode;
    maxPyramidNode.put_value(max_allowed_pyramids);
    max_num_pyramids_ptree.push_back(
        std::make_pair("numPyramids", maxPyramidNode));
    max_num_pyramids_ptree.add_child(
        "Conservativeness" + std::to_string(max_allowed_pyramids),
        conservativenesss_ptree);

    root.put_child("MaxNumPyramids", max_num_pyramids_ptree);  // put_child overwrites, unlike add_child.

    std::ofstream json_out("./data/conservativenessTest.json");
    write_json(json_out, root);
    json_out.close();
  }
}

static void run_collision_checking_time_benchmark(
    const camera_intrinsics_t& camera_intrinsics,
    const planner_specs_t& planner_specs,
    const synthesis_specs_t& synthesis_specs,
    const benchmark_options_t& benchmark_options,
    const collision_time_options_t& collision_time_options,
    boost::property_tree::ptree& root) {

  if (benchmark_options.display) {
    cv::namedWindow("display", cv::WINDOW_AUTOSIZE);
  }

  std::mt19937 image_rdgen(synthesis_specs.random_seed);
  std::mt19937 state_rdgen(benchmark_options.random_seed);
  std::uniform_real_distribution<> random_vx(benchmark_options.vx_min,
                                             benchmark_options.vx_max);
  std::uniform_real_distribution<> random_vy(benchmark_options.vy_min,
                                             benchmark_options.vy_max);
  std::uniform_real_distribution<> random_vz(benchmark_options.vz_min,
                                             benchmark_options.vz_max);
  std::uniform_real_distribution<> random_ay(benchmark_options.ay_min,
                                             benchmark_options.ay_max);
  std::vector<double> collision_check_time;
  std::vector<int> num_pyramids;
  for (int i = 0; i < synthesis_specs.num_scenes; i++) {
    if (i % 20 == 0) {
      printf("Processed %d frames so far.\n", i);
    }

    cv::Mat depth_mat;
    make_image(camera_intrinsics, synthesis_specs, image_rdgen, depth_mat);
    DepthImagePlanner planner(depth_mat, camera_intrinsics.depth_scale,
                              camera_intrinsics.f, camera_intrinsics.cx,
                              camera_intrinsics.cy,
                              planner_specs.physical_vehicle_radius,
                              planner_specs.vehicle_radius_for_planning,
                              planner_specs.minimum_collision_distance);

    RapidTrajectoryGenerator traj(
        Vec3(0, 0, 0),
        Vec3(random_vx(state_rdgen), random_vy(state_rdgen),
             random_vz(state_rdgen)),
        Vec3(0, -random_ay(state_rdgen), 0), Vec3(0, 9.81, 0));

    double total_check_time;
    planner.MeasureCollisionCheckingSpeed(
        collision_time_options.num_traj,
        collision_time_options.pyramid_gen_time, traj, total_check_time);

    collision_check_time.push_back(
        total_check_time / collision_time_options.num_traj);
    num_pyramids.push_back(planner.GetPyramids().size());

    if (benchmark_options.png_output) {
      std::string filename = std::to_string(i) + ".png";
      imwrite(filename, depth_mat);
    }
    if (benchmark_options.display) {
      display_image(
          depth_mat,
          benchmark_options.display_range_min / camera_intrinsics.depth_scale,
          benchmark_options.display_range_max / camera_intrinsics.depth_scale);
    }
  }

  benchmark_stats_t stats_coll_check_time = calculate_stats<double>(
      collision_check_time);
  root.put("AvgCollCheckTimePerTrajNs", stats_coll_check_time.mean);

  benchmark_stats_t stats_num_pyramids = calculate_stats<int>(num_pyramids);
  root.put("AvgPyramidsGen", stats_num_pyramids.mean);

  printf("Avg. Collision checking time per traj. = %f ns\n",
         stats_coll_check_time.mean);
  printf("Avg. pyramids per frame = %f\n", stats_num_pyramids.mean);

  print_stats(std::cout, "AvgCollCheckTimePerTrajNs", stats_coll_check_time);
  print_stats(std::cout, "AvgPyramidsGen", stats_num_pyramids);

  std::ofstream json_out("./data/collisionCheckingTime.json");
  write_json(json_out, root);
  json_out.close();
}

static void run_trajectory_coverage_benchmark(
    const camera_intrinsics_t& camera_intrinsics,
    const planner_specs_t& planner_specs,
    const synthesis_specs_t& synthesis_specs,
    const benchmark_options_t& benchmark_options,
    const traj_coverage_options_t& traj_coverage_options,
    boost::property_tree::ptree& root) {

  if (benchmark_options.display) {
    cv::namedWindow("display", cv::WINDOW_AUTOSIZE);
  }

  std::mt19937 image_rdgen(synthesis_specs.random_seed);
  std::mt19937 state_rdgen(benchmark_options.random_seed);
  std::uniform_real_distribution<> random_vx(benchmark_options.vx_min,
                                             benchmark_options.vx_max);
  std::uniform_real_distribution<> random_vy(benchmark_options.vy_min,
                                             benchmark_options.vy_max);
  std::uniform_real_distribution<> random_vz(benchmark_options.vz_min,
                                             benchmark_options.vz_max);
  std::uniform_real_distribution<> random_ay(benchmark_options.ay_min,
                                             benchmark_options.ay_max);

  boost::property_tree::ptree comp_time_ptree, avg_traj_gen_ptree;
  for (int k = 0; k < traj_coverage_options.num_comp_times; k++) {
    double a = traj_coverage_options.min_comp_time;
    double b = traj_coverage_options.max_comp_time;
    double n = traj_coverage_options.num_comp_times;
    double base = pow(b / a, 1 / (n - 1));  // This increasing computation time geometrically, so we have more resolution at smaller computation times
    double comp_time = a * pow(base, k);  // When k == n-1, compTime = b
    std::vector<int> num_traj_gen;
    for (int i = 0; i < synthesis_specs.num_scenes; i++) {
      if (i % 20 == 0) {
        printf("Processed %d frames so far.\n", i);
      }
      cv::Mat depth_mat;
      make_image(camera_intrinsics, synthesis_specs, image_rdgen, depth_mat);
      DepthImagePlanner planner(depth_mat, camera_intrinsics.depth_scale,
                                camera_intrinsics.f, camera_intrinsics.cx,
                                camera_intrinsics.cy,
                                planner_specs.physical_vehicle_radius,
                                planner_specs.vehicle_radius_for_planning,
                                planner_specs.minimum_collision_distance);

      RapidTrajectoryGenerator traj(
          Vec3(0, 0, 0),
          Vec3(random_vx(state_rdgen), random_vy(state_rdgen),
               random_vz(state_rdgen)),
          Vec3(0, -random_ay(state_rdgen), 0), Vec3(0, 9.81, 0));

      Vec3 exploration_direction(0, 0, 1);
      planner.FindFastestTrajRandomCandidates(traj, comp_time,
                                              exploration_direction);

      num_traj_gen.push_back(planner.GetNumTrajectoriesGenerated());

      if (benchmark_options.png_output) {
        std::string filename = std::to_string(i) + ".png";
        imwrite(filename, depth_mat);
      }
      if (benchmark_options.display) {
        display_image(
            depth_mat,
            benchmark_options.display_range_min / camera_intrinsics.depth_scale,
            benchmark_options.display_range_max
                / camera_intrinsics.depth_scale);
      }
    }

    benchmark_stats_t stats_num_traj_gen = calculate_stats<int>(num_traj_gen);

    printf("Stats for %f s of computation time:", comp_time);
    print_stats(std::cout, "numTrajGen", stats_num_traj_gen);

    boost::property_tree::ptree comp_time_node;
    comp_time_node.put_value(comp_time);
    comp_time_ptree.push_back(std::make_pair("", comp_time_node));

    boost::property_tree::ptree avg_traj_gen_node;
    avg_traj_gen_node.put_value(stats_num_traj_gen.mean);
    avg_traj_gen_ptree.push_back(std::make_pair("", avg_traj_gen_node));

    root.put_child("CompTime", comp_time_ptree);  // * put_child replaces the node, instead of adding a node like add_child.
    root.put_child("AvgTrajGen", avg_traj_gen_ptree);

    std::ofstream json_out("./data/avgTrajGen.json");
    write_json(json_out, root);
    json_out.close();
  }
}

int main(int argc, const char* argv[]) {
  using namespace boost::program_options;
  using namespace boost::property_tree;

  try {
    camera_intrinsics_t camera_intrinsics { };
    planner_specs_t planner_specs { };
    synthesis_specs_t synthesis_specs { };
    benchmark_options_t benchmark_options { };

    conservative_test_options_t conservative_test_options { };
    collision_time_options_t collision_time_options { };
    traj_coverage_options_t traj_coverage_options { };

    options_description desc { "Options" };
    options_description_easy_init opt = desc.add_options();
    opt("help,h", "Help screen");

    opt("test_type",
        value<int>(&benchmark_options.test_type)->default_value(
            TRAJECTORY_COVERAGE),
        "test to be run (CONSERVATIVENESS = 0, COLLISION_CHECKING_TIME = 1, TRAJECTORY_COVERAGE = 2)");

    opt("image-count,n",
        value<uint>(&synthesis_specs.num_scenes)->default_value(500),
        "number of scenes for synthetic depth images");
    opt("w", value<int>(&camera_intrinsics.w)->default_value(160),
        "synthesized image width");
    opt("h", value<int>(&camera_intrinsics.h)->default_value(120),
        "synthesized image height");
    opt("depth-scale",
        value<double>(&camera_intrinsics.depth_scale)->default_value(0.001),
        "depth scale");
    opt("f", value<double>(&camera_intrinsics.f)->default_value(386.643 / 4.0),
        "synthesized image focal length");  // This has to be scaled when images scaled down.
    opt("cx", value<double>(&camera_intrinsics.cx)->default_value(160.0 / 2.0),
        "synthesized image cx");
    opt("cy", value<double>(&camera_intrinsics.cy)->default_value(120.0 / 2.0),
        "synthesized image cy");

    opt("vehicleRadius",
        value<double>(&planner_specs.physical_vehicle_radius)->default_value(
            0.26),
        "The true physical radius of the vehicle (ignore depth values closer than this) [meters]");
    opt("planningRadius",
        value<double>(&planner_specs.vehicle_radius_for_planning)->default_value(
            0.46),
        "The radius of the vehicle used for planning [meters]");
    opt("minimumCollisionDistance",
        value<double>(&planner_specs.minimum_collision_distance)->default_value(
            1.0),
        "We assume on obstacle is just outside the field of view this distance away [meters]");

    opt("obs-num",
        value<uint>(&synthesis_specs.num_obstacles)->default_value(2),
        "number of obstacles for synthetic depth images");
    opt("obs-w-min",
        value<double>(&synthesis_specs.obstacles_width_min)->default_value(
            0.20),
        "min width of obstacles in [m]");
    opt("obs-w-max",
        value<double>(&synthesis_specs.obstacles_width_max)->default_value(
            0.20),
        "max width of obstacles in [m]");
    opt("obs-d-min",
        value<double>(&synthesis_specs.obstacles_depth_min)->default_value(1.5),
        "min depth of obstacles in [m]");
    opt("obs-d-max",
        value<double>(&synthesis_specs.obstacles_depth_max)->default_value(3.0),
        "max depth of obstacles in [m]");
    opt("obs-seed", value<uint>(&synthesis_specs.random_seed)->default_value(0),
        "random seed for generating images");

    opt("dispaly,d",
        bool_switch(&benchmark_options.display)->default_value(false),
        "display each frame");
    opt("display-min",
        value<double>(&benchmark_options.display_range_min)->default_value(0.0),
        "min of depth range for display");
    opt("display-max",
        value<double>(&benchmark_options.display_range_max)->default_value(4.0),
        "max of depth range for display");
    opt("png-output,o",
        bool_switch(&benchmark_options.png_output)->default_value(false),
        "output png files");
    opt("vx-min", value<double>(&benchmark_options.vx_min)->default_value(-1),
        "min vx [m/s]");
    opt("vx-max", value<double>(&benchmark_options.vx_max)->default_value(1),
        "max vx [m/s]");
    opt("vy-min", value<double>(&benchmark_options.vy_min)->default_value(-1),
        "min vy [m/s]");
    opt("vy-max", value<double>(&benchmark_options.vy_max)->default_value(1),
        "max vy [m/s]");
    opt("vz-min", value<double>(&benchmark_options.vz_min)->default_value(0),
        "min vz [m/s]");
    opt("vz-max", value<double>(&benchmark_options.vz_max)->default_value(4),
        "max vz [m/s]");
    opt("ay-min", value<double>(&benchmark_options.ay_min)->default_value(-5),
        "min ax [m/s]");
    opt("ay-max", value<double>(&benchmark_options.ay_max)->default_value(5),
        "max ay [m/s]");
    opt("bc-seed",
        value<uint>(&benchmark_options.random_seed)->default_value(0),
        "random seed for initial states");

    opt("maxNumPyramidForConservativenessTest",
        value<std::vector<int>>(&conservative_test_options.max_num_pyramids)
            ->multitoken()->default_value(std::vector<int> { 1, 2, 4, 8 },
                                          "1 2 4 8"),
        "List of the maximum allowed number of pyramids (for use with CONSERVATIVENESS only)");
    opt("numTrajForConservativenessTest",
        value<int>(&conservative_test_options.num_traj)->default_value(1000),
        "Number of trajectories to use to compute conservativeness (for use with CONSERVATIVENESS only)");

    opt("pyramidGenTimeForCCTest",
        value<double>(&collision_time_options.pyramid_gen_time)->default_value(
            0.00181),
        "Time allocated for generating pyramids (for use with COLLISION_CHECKING_TIME only)");
    opt("numTrajForCCTest",
        value<int>(&collision_time_options.num_traj)->default_value(10000),
        "Number of trajectories to use to time collision checker (for use with COLLISION_CHECKING_TIME only)");

    opt("minCompTimeForTCTest",
        value<double>(&traj_coverage_options.min_comp_time)->default_value(
            0.0001),
        "Minimum computation time allocated to planner (for use with TRAJECTORY_COVERAGE only)");
    opt("maxCompTimeForTCTest",
        value<double>(&traj_coverage_options.max_comp_time)->default_value(
            0.05),
        "Maximum computation time allocated to planner (for use with TRAJECTORY_COVERAGE only)");
    opt("numCompTimesForTCTest",
        value<int>(&traj_coverage_options.num_comp_times)->default_value(10),
        "Number of computation times to test between min and max (for use with TRAJECTORY_COVERAGE only)");

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if (vm.count("help")) {
      std::cout << "\n" << desc << "\n";
      return 0;
    }

    std::cout
        << std::left << std::endl
        << "* please make sure the executable is compiled with optimization."
        << std::endl << "* please make sure the CPU governor is performance."
        << std::endl;

    // std::vector<cv::Mat> qDepthFrames;

    ptree root;

    // * each item will be structured as per dot's
    root.put("synthesis_specs.camera_intrinsics.w", camera_intrinsics.w);
    root.put("synthesis_specs.camera_intrinsics.h", camera_intrinsics.h);
    root.put("synthesis_specs.camera_intrinsics.depth_scale",
             camera_intrinsics.depth_scale);
    root.put("synthesis_specs.camera_intrinsics.f", camera_intrinsics.f);
    root.put("synthesis_specs.camera_intrinsics.cx", camera_intrinsics.cx);
    root.put("synthesis_specs.camera_intrinsics.cy", camera_intrinsics.cy);
    root.put("synthesis_specs.num_scenes", synthesis_specs.num_scenes);
    root.put("synthesis_specs.num_obstacles", synthesis_specs.num_obstacles);
    root.put("synthesis_specs.obstacles_width_min",
             synthesis_specs.obstacles_width_min);
    root.put("synthesis_specs.obstacles_width_max",
             synthesis_specs.obstacles_width_max);
    root.put("synthesis_specs.obstacles_depth_min",
             synthesis_specs.obstacles_depth_min);
    root.put("synthesis_specs.obstacles_depth_max",
             synthesis_specs.obstacles_depth_max);
    root.put("synthesis_specs.random_seed", synthesis_specs.random_seed);

    root.put("benchmark_options.display", benchmark_options.display);
    root.put("benchmark_options.display_range_min",
             benchmark_options.display_range_min);
    root.put("benchmark_options.display_range_max",
             benchmark_options.display_range_max);
    root.put("benchmark_options.png_output", benchmark_options.png_output);
    root.put("benchmark_options.vx_min", benchmark_options.vx_min);
    root.put("benchmark_options.vx_max", benchmark_options.vx_max);
    root.put("benchmark_options.vy_min", benchmark_options.vy_min);
    root.put("benchmark_options.vy_max", benchmark_options.vy_max);
    root.put("benchmark_options.vz_min", benchmark_options.vz_min);
    root.put("benchmark_options.vz_max", benchmark_options.vz_max);
    root.put("benchmark_options.ax_min", benchmark_options.ay_min);
    root.put("benchmark_options.ax_max", benchmark_options.ay_max);
    root.put("benchmark_options.random_seed", benchmark_options.random_seed);

    // Run benchmark and save data
    try {
      if (benchmark_options.test_type == CONSERVATIVENESS) {
        // const properties saved first
        ptree child;
        for (auto &x : conservative_test_options.max_num_pyramids) {
          ptree node;
          node.put_value(x);
          child.push_back(std::make_pair("", node));
        }
        root.add_child("conservative_test_options.max_num_pyramids", child);
        root.put("conservative_test_options.num_traj",
                 conservative_test_options.num_traj);

        run_conservativeness_benchmark(camera_intrinsics, planner_specs,
                                       synthesis_specs, benchmark_options,
                                       conservative_test_options, root);
      } else if (benchmark_options.test_type == COLLISION_CHECKING_TIME) {
        // const properties saved first
        root.put("collision_time_options.pyramid_gen_time",
                 collision_time_options.pyramid_gen_time);
        root.put("collision_time_options.num_traj",
                 collision_time_options.num_traj);

        run_collision_checking_time_benchmark(camera_intrinsics, planner_specs,
                                              synthesis_specs,
                                              benchmark_options,
                                              collision_time_options, root);

      } else if (benchmark_options.test_type == TRAJECTORY_COVERAGE) {
        // const properties saved first
        root.put("traj_coverage_options.min_comp_time",
                 traj_coverage_options.min_comp_time);
        root.put("traj_coverage_options.max_comp_time",
                 traj_coverage_options.max_comp_time);
        root.put("traj_coverage_options.num_comp_times",
                 traj_coverage_options.num_comp_times);

        run_trajectory_coverage_benchmark(camera_intrinsics, planner_specs,
                                          synthesis_specs, benchmark_options,
                                          traj_coverage_options, root);
      } else {
        printf("Invalid choice of benchmarking test to run. Aborting.\n");
      }
    } catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
    }

  } catch (const error& ex) {
    std::cerr << ex.what() << '\n';
  }

  return 0;
}
