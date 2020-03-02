# Rectangular Pyramid Partitioning using Integrated Depth Sensors (RAPPIDS)

This repository contains source code implementing an algorithm for quickly finding local collision-free trajectories given a single depth image from an onboard camera. The algorithm leverages a new pyramid-based spatial partitioning method that enables rapid collision detection between candidate trajectories and the environment. Due to its efficiency, the algorithm can be run at high rates on computationally constrained hardware, evaluating thousands of candidate trajectories in milliseconds.

The algorithm is described in a paper submitted to IEEE Robotics and Automation Letters (RA-L) with the International Conference on Intelligent Robots and Systems 2020 (IROS) option. A preprint version of the paper will be made available soon.

Contact: Nathan Bucki (nathan_bucki@berkeley.edu)
High Performance Robotics Lab, Dept. of Mechanical Engineering, UC Berkeley

## Getting Started

First clone the repository and enter the created folder:
```
git clone https://github.com/nlbucki/RAPPIDS.git
cd RAPPIDS
```

Create a build folder and compile:
```
mkdir build
cd build
cmake ..
make
```

A program is provided that demonstrates the performance of the algorithm and gives an example of how the algorithm can be used to generate collision free motion primitives. The program `Benchmarker` performs the Monte Carlo simulations described in Seciton IV of the associated paper. The three tests performed in the paper can be ran from the RAPPIDS folder with the following commands:
```
./build/test/Benchmarker --test_type 0 -n 10000 --maxNumPyramidForConservativenessTest 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
./build/test/Benchmarker --test_type 1 -n 10000 
./build/test/Benchmarker --test_type 2 -n 10000 --w 640 --h 480 --f 386 --cx 320 --cy 240 --numCompTimesForTCTest 20
```
Note the `-n` option can be changed to a smaller number to perform less Monte Carlo trials, and thus run the tests faster (but with less accuracy). The above settings reflect those used to generate the results reported in the paper.

Each test generates a `.json` file in the data folder containing the test results. We provide two python scripts to visualize the results of the conservativeness test (test 0) and the overall planner performance test (test 2). They can be ran with the following commands:
```
cd scripts
python plotAvgTrajGenNum.py
python plotConservativeness.py
```

## Documentation
An HTML file generated with [Doxygen](http://www.doxygen.nl/) can be accessed after cloning the repository by opening `Documentation.html` in the `doc/` folder.

## Licensing

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
