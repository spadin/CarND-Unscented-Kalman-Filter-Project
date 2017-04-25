#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.7;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;

  n_x_ = 5;

  n_aug_ = 7;

  lambda_ = 3 - n_aug_;

	// set weights
	weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);

	float w_2_plus = 1 / (2 * (lambda_ + n_aug_));
	for(int j = 1; j < 2 * n_aug_ + 1; j++) {
    weights_(j) = w_2_plus;
	}
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if(!is_initialized_) {
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;

    previous_timestamp_ = meas_package.timestamp_;

    if(use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
      is_initialized_ = true;
      return;
    }

    if(use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      VectorXd cartesian = PolarToCartesian(meas_package.raw_measurements_);
      x_ << cartesian[0], cartesian[1], meas_package.raw_measurements_[2], 0, 0;
      is_initialized_ = true;
      return;
    }
  }

  long current_timestamp = meas_package.timestamp_;
  float delta_t = (current_timestamp - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = meas_package.timestamp_;

  Prediction(delta_t);

  if(use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }

  if(use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
}

VectorXd UKF::PolarToCartesian(const VectorXd& x_state) {
  float rho = x_state[0];
  float phi = x_state[1];
  float rho_dot = x_state[2];

  float px = rho * cos(phi);
  float py = rho * sin(phi);
  float vx = rho_dot * cos(phi);
  float vy = rho_dot * sin(phi);

  VectorXd cartesian(4);
  cartesian << px,
               py,
               vx,
               vy;

  return cartesian;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  MatrixXd Xsig_aug = MatrixXd(15, 7);
  Xsig_aug = AugmentedSigmaPoints();
  SigmaPointPrediction(Xsig_aug, delta_t);
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  //set measurement dimension, lidar can measure px, py
  int n_z = 2;

	// create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);

	// mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

	// measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  // transform sigma points into measurement space
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);

    Zsig(0,i) = p_x;
    Zsig(1,i) = p_y;
  }

  // calculate mean predicted measurement
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // calculate measurement covariance matrix S
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S(0,0) += std_laspx_ * std_laspx_;
  S(1,1) += std_laspy_ * std_laspy_;

	// create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

	// calculate cross correlation matrix
	for(int i = 0; i < 2 * n_aug_ + 1; i++) {
		// residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

  // calculate Kalman gain K;
  MatrixXd Kg = Tc * S.inverse();

  // set z to raw measurements
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_diff = z - z_pred;

  // update state mean and covariance matrix
  x_ = x_ + Kg * z_diff;
  P_ = P_ - Kg *  S * Kg.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

	// create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);

	// mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

	// measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);

  // transform sigma points into measurement space
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double psi = Xsig_pred_(3,i);

    Zsig(0,i) = sqrt(p_x * p_x + p_y * p_y);
    Zsig(1,i) = atan(p_y / p_x);
    Zsig(2,i) = (p_x * cos(psi) * v + p_y * sin(psi) * v) / Zsig(0,i);

  }

  // calculate mean predicted measurement
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // calculate measurement covariance matrix S
  for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S(0,0) += std_radr_ * std_radr_;
  S(1,1) += std_radphi_ * std_radphi_;
  S(2,2) += std_radrd_ * std_radrd_;

	// create matrix for cross correlation Tc
	MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

	// calculate cross correlation matrix
	for(int i = 0; i < 2 * n_aug_ + 1; i++) {
		// residual
		VectorXd z_diff = Zsig.col(i) - z_pred;

		// angle normalization
    z_diff(1) = NormalizeAngle(z_diff(1));

		// state difference
		VectorXd x_diff = Xsig_pred_.col(i) - x_;

		// angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

		Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
	}

  // calculate Kalman gain K;
  MatrixXd Kg = Tc * S.inverse();

  // set z to raw measurements
  VectorXd z = meas_package.raw_measurements_;

  VectorXd z_diff = z - z_pred;
  z_diff(1) = NormalizeAngle(z_diff(1));

  // update state mean and covariance matrix
  x_ = x_ + Kg * z_diff;
  P_ = P_ - Kg *  S * Kg.transpose();
}

MatrixXd UKF::AugmentedSigmaPoints() {
  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.fill(0.0);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.fill(0.0);

  // create augmented mean state
  x_aug << x_, 0, 0;

  // create augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) << P_;
  P_aug.bottomRightCorner(2, 2) << (std_a_ * std_a_), 0, 0, (std_yawdd_ * std_yawdd_);

	// create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  //create augmented sigma points
  MatrixXd first_section = x_aug;

  double sqrt_lambda = sqrt(lambda_ + n_aug_);
  MatrixXd lambda_times_A = sqrt_lambda * A;

  MatrixXd x_aug_spread(n_aug_, n_aug_);
  x_aug_spread << x_aug, x_aug, x_aug, x_aug, x_aug, x_aug, x_aug;

  MatrixXd second_section = x_aug_spread + lambda_times_A;
  MatrixXd third_section = x_aug_spread - lambda_times_A;

  Xsig_aug << first_section, second_section, third_section;
  // std::cout << "Xsig_aug" << std::endl;
  // std::cout << Xsig_aug << std::endl;

  return Xsig_aug;
}


void UKF::SigmaPointPrediction(MatrixXd Xsig_aug, double delta_t) {
	// create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //predict sigma points
  //avoid division by zero
  //write predicted sigma points into right column
  for(int i = 0; i < (2 * n_aug_ + 1); i++) {
    float px = Xsig_aug(0,i);
    float py = Xsig_aug(1,i);
    float v = Xsig_aug(2,i);
    float psi = Xsig_aug(3,i);
    float psi_dot = Xsig_aug(4,i);
    float nu_a = Xsig_aug(5,i);
    float nu_psi_dot_dot = Xsig_aug(6,i);

    float delta_t_squared = delta_t * delta_t;
    float sin_psi = sin(psi);
    float cos_psi = cos(psi);


    if(fabs(psi_dot) < 0.001) {
      VectorXd pred(n_x_);
      pred << px + v * cos_psi * delta_t + 0.5 * delta_t_squared * cos_psi * nu_a,
              py + v * sin_psi * delta_t + 0.5 * delta_t_squared * sin_psi * nu_a,
              v + 0 + delta_t * nu_a,
              psi + psi_dot * delta_t + 0.5 * delta_t_squared * nu_psi_dot_dot,
              psi_dot + 0 + delta_t * nu_psi_dot_dot;

      Xsig_pred.col(i) = pred;
    }
    else {
      float v_over_psi_dot = v / psi_dot;

      VectorXd pred(n_x_);
      pred << px + v_over_psi_dot * (sin(psi + psi_dot * delta_t) - sin_psi) + 0.5 * delta_t_squared * cos_psi * nu_a,
              py + v_over_psi_dot * (-cos(psi + psi_dot * delta_t) + cos_psi) + 0.5 * delta_t_squared * sin_psi * nu_a,
              v + 0 + delta_t * nu_a,
              psi + psi_dot * delta_t + 0.5 * delta_t_squared * nu_psi_dot_dot,
              psi_dot + 0 + delta_t * nu_psi_dot_dot;

      Xsig_pred.col(i) = pred;
    }
  }

  Xsig_pred_ = Xsig_pred;
}

void UKF::PredictMeanAndCovariance() {
  // create vector for predicted state
  VectorXd x = VectorXd(n_x_);
  x.fill(0.0);

  // create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);

	// predict state mean
	for(int i = 0; i < 2 * n_aug_ + 1; i++) {
		x = x + weights_(i) * Xsig_pred_.col(i);
	}

	// predict state covariance matrix
	for(int i = 0; i < 2 * n_aug_ + 1; i++) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x;

    // angle normalization
    x_diff(3) = NormalizeAngle(x_diff(3));

		P = P + weights_(i) * x_diff * x_diff.transpose();
	}

  x_ = x;
  P_ = P;
}

double UKF::NormalizeAngle(double angle) {
  return atan2(sin(angle), cos(angle));
}
