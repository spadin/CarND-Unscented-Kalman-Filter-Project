#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */

  VectorXd rmse(4);
  rmse.fill(0.0);

  if(estimations.size() == 0) {
    std::cout << "Error, estimations size is 0" << std::endl;
    return rmse;
  }

  if(estimations.size() != ground_truth.size()) {
    std::cout << "Error, estimations and ground_truth are not the same size" << std::endl;
    return rmse;
  }

  VectorXd residual(4);
  //accumulate squared residuals
  for(int i = 0; i < estimations.size(); i++){
    residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse /= estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}
