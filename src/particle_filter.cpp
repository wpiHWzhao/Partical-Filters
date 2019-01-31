/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <assert.h>

#include "helper_functions.h"

using std::string;
using std::vector;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  std::default_random_engine gen;

  num_particles = 1024;  // TODO: Set the number of particles

  double x_std = std[0];
  double y_std = std[1];
  double theta_std = std[2];

  weights = std::vector<double>(static_cast<unsigned long>(num_particles), 1.0);

  std::normal_distribution<double> x_dist(x,x_std);
  std::normal_distribution<double> y_dist(y,y_std);
  std::normal_distribution<double> theta_dist(theta,theta_std);


  for(int i=0; i < num_particles; ++i){
    Particle parts;
    parts.id = i;
    parts.x = x_dist(gen);
    parts.y = y_dist(gen);
    parts.theta = theta_dist(gen);
    parts.weight = 1.0;
    particles.push_back(parts);
  }

  is_initialized = true;


}


void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
   std::normal_distribution<double> x_dist(0,std_pos[0]);
   std::normal_distribution<double> y_dist(0,std_pos[1]);
   std::normal_distribution<double> theta_dist(0,std_pos[2]);
   std::default_random_engine gen;

   for(int i=0; i < num_particles; ++i){
     if (fabs(yaw_rate)>0.00001){
       particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
       particles[i].y += (velocity/yaw_rate)*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
       particles[i].theta += yaw_rate*delta_t;
     } else{
       particles[i].x += velocity*delta_t*cos(particles[i].theta);
       particles[i].y += velocity*delta_t*sin(particles[i].theta);
     }

     particles[i].x += x_dist(gen);
     particles[i].y += y_dist(gen);
     particles[i].theta += theta_dist(gen);
   }

}


void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
   for(int i = 0; i < observations.size(); ++i){
     double minDist = std::numeric_limits<double >::max();
     observations[i].id = -1;
     for(int j =0; j < predicted.size(); ++j){
       double Dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
       if(Dist<minDist){
         minDist = Dist;
         observations[i].id = predicted[j].id;
       }
     }
     assert(observations[i].id != -1);
   }

}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */


   for(int i = 0; i < num_particles; ++i){
     vector<LandmarkObs> withinRangeLM;// The landmarks that are within range.
     vector<LandmarkObs> obsInWorldCord;// The observations that are transformed into world coordinate
     for(int j =0; j < map_landmarks.landmark_list.size();++j){
       if (dist(map_landmarks.landmark_list[j].x_f,map_landmarks.landmark_list[j].y_f,particles[i].x,particles[i].y) <=
       sensor_range){
         withinRangeLM.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i,map_landmarks.landmark_list[j].x_f,
                                             map_landmarks.landmark_list[j].y_f});
       }
     }

     // assert(!withinRangeLM.empty());

     for(int j = 0; j < observations.size(); ++j){
       double x_trans = particles[i].x+(cos(particles[i].theta)*observations[j].x)-(sin(particles[i].theta)*
               observations[j].y);
       double y_trans = particles[i].y+(sin(particles[i].theta)*observations[j].x)+(cos(particles[i].theta)*
               observations[j].y);
       obsInWorldCord.push_back(LandmarkObs{-1,x_trans,y_trans});
     }

     dataAssociation(withinRangeLM,obsInWorldCord);
     particles[i].weight = 1.0;
     for(int j = 0; j < obsInWorldCord.size(); ++j){
       for(int k =0; k < withinRangeLM.size(); ++k){
         // Search for the landmark with same ID
         if (withinRangeLM[k].id == obsInWorldCord[j].id){
           particles[i].weight *= multiv_prob(std_landmark[0],std_landmark[1],obsInWorldCord[j].x,obsInWorldCord[j].y,
                   withinRangeLM[k].x,withinRangeLM[k].y);
           break;
         }
       }
     }
     weights[i]=particles[i].weight;

   }
   // assert(!weights.empty());



}



void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
   std::default_random_engine gen;

   std::discrete_distribution<> distribution(weights.begin(),weights.end());

   vector<Particle> resampledParticle;

   // assert(!particles.empty());
   // std::cout<<particles.size()<<std::endl;

   for (int i =0 ; i< particles.size(); ++i){
     resampledParticle.push_back(particles[distribution(gen)]);
   }

   // assert(!resampledParticle.empty());

   particles = resampledParticle;



}


void ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                         const std::vector<double> &sense_x, const std::vector<double> &sense_y)
{
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}


string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}